"""
CogniFin AI - LLM Client (Google GenAI SDK)
===========================================
Unified LLM Client powered by Google's official GenAI SDK (google-genai).
Provides deterministic generation, multi-query expansion, and structured JSON parsing.

Zero raw HTTP calls — driven entirely by official Google SDK client methods.
"""

import os
import json
import logging
from typing import Optional, Dict, Any

try:
    from google import genai
    from google.genai import types as genai_types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger("cognifin.rag.llm")


class LLMClient:
    """
    Unified LLM Client using Google GenAI SDK.
    Unified LLM & Embedding Client using Google GenAI SDK and FastEmbed ONNX.
    """

    def __init__(self):
        self.api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or getattr(settings, "LLM_API_KEY", "")
            or getattr(settings, "GEMINI_API_KEY", "")
            or ""
        )
        self.model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GEMINI_MODEL")
            or getattr(settings, "LLM_MODEL", "")
            or getattr(settings, "GEMINI_MODEL", "")
            or "gemini-3.6-flash"
        )
        self.embedding_model = getattr(settings, "EMBEDDING_MODEL", "gemini-embedding-001")
        self.embedding_model = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.embedding_dim = int(getattr(settings, "EMBEDDING_DIM", 768))

        self.is_configured = bool(
            self.api_key
            and not self.api_key.startswith("your_")
            and GOOGLE_GENAI_AVAILABLE
        )

        self.client: Optional[genai.Client] = None
        self._fastembed_model: Optional[Any] = None

        if FASTEMBED_AVAILABLE:
            try:
                self._fastembed_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
                logger.info("FastEmbed ONNX engine loaded (BAAI/bge-base-en-v1.5, 768-dim, local CPU)")
            except Exception as fe_err:
                logger.warning(f"Could not initialize FastEmbed: {fe_err}")

        if self.is_configured:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"LLM Client initialized with Google GenAI SDK (model: {self.model})")
                logger.info(f"LLM Client initialized with Google GenAI SDK (model: {self.model}, embed: {self.embedding_model})")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {e}")
                self.is_configured = False
        else:
            logger.info("LLM API key not configured — /retrieve and /health will operate normally.")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a 768-dim float vector using FastEmbed ONNX (instant, free).
        """
        if self._fastembed_model is not None:
            try:
                emb = list(self._fastembed_model.embed([text]))[0]
                return emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception as e:
                logger.warning(f"FastEmbed failed ({e}), falling back to API...")

        if not self.is_configured or self.client is None:
            raise ValueError("No embedding engine available (FastEmbed not loaded and API key missing).")

        try:
            config = genai_types.EmbedContentConfig(output_dimensionality=self.embedding_dim)
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=config,
            )
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            raise ValueError("Empty embedding returned from Google GenAI API")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings into normalized float vectors.
        """
        if self._fastembed_model is not None:
            try:
                embeddings = list(self._fastembed_model.embed(texts, batch_size=256))
                return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
            except Exception as e:
                logger.warning(f"FastEmbed batch failed ({e}), falling back to API...")

        if not self.is_configured or self.client is None:
            raise ValueError("No embedding engine available.")
        if not texts:
            return []

        try:
            config = genai_types.EmbedContentConfig(output_dimensionality=self.embedding_dim)
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=config,
            )
            if response.embeddings:
                return [e.values for e in response.embeddings]
            raise ValueError("Empty embeddings returned from Google GenAI API")
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise RuntimeError(f"Error generating batch embeddings: {str(e)}")


    def stream_generate(self, system_prompt: str, user_message: str, temperature: float = 0.0):
        """
        Yield real-time tokens from Gemini as they are generated.
        """
        if not self.is_configured or self.client is None:
            raise ValueError("LLM API key is not configured.")

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
            max_output_tokens=2000,
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=user_message,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    def _get_candidate_models(self) -> list:
        candidates = [self.model, "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        seen = set()
        unique_candidates = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                unique_candidates.append(m)
        return unique_candidates

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        """
        Generate a grounded answer using Google GenAI SDK with graceful fallback across candidate models.
        """
        if not self.is_configured or self.client is None:
            raise ValueError(
                "LLM API key is not configured. "
                "Please set GEMINI_API_KEY or LLM_API_KEY in your .env file and restart the server."
            )

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
            max_output_tokens=1500,
        )

        last_err = None
        for model_name in self._get_candidate_models():
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=config,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_err = e
                logger.warning(f"Model {model_name} failed ({e}), attempting fallback...")

        logger.error(f"All LLM generation candidates failed: {last_err}")
        raise RuntimeError(f"Error calling LLM provider: {str(last_err)}")

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Simple text generation for multi-query expansion and prompts.
        """
        return self.generate(system_prompt="", user_message=prompt, temperature=temperature)

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate and parse structured JSON output with candidate model fallback.
        """
        if not self.is_configured or self.client is None:
            raise ValueError("LLM API key is not configured.")

        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            response_mime_type="application/json",
        )

        last_err = None
        for model_name in self._get_candidate_models():
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw = response.text.strip() if response and response.text else "{}"
                return json.loads(raw)
            except Exception as e:
                last_err = e
                logger.warning(f"Model {model_name} JSON generation failed ({e}), attempting fallback...")

        logger.error(f"All LLM JSON generation candidates failed: {last_err}")
        raise RuntimeError(f"Error generating structured output: {str(last_err)}")

