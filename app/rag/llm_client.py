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

logger = logging.getLogger("cognifin.rag.llm")


class LLMClient:
    """
    Unified LLM Client using Google GenAI SDK.
    """

    def __init__(self):
        # Support LLM_API_KEY or GEMINI_API_KEY
        self.api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or ""
        )
        self.model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GEMINI_MODEL")
            or "gemini-2.0-flash"
        )

        self.is_configured = bool(
            self.api_key
            and not self.api_key.startswith("your_")
            and GOOGLE_GENAI_AVAILABLE
        )

        self.client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"LLM Client initialized with Google GenAI SDK (model: {self.model})")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {e}")
                self.is_configured = False
        else:
            logger.info("LLM API key not configured — /retrieve and /health will operate normally.")

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        """
        Generate a grounded answer using Google GenAI SDK.
        """
        if not self.is_configured or self.client is None:
            raise ValueError(
                "LLM API key is not configured. "
                "Please set GEMINI_API_KEY or LLM_API_KEY in your .env file and restart the server."
            )

        try:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=1500,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=config,
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"Error calling LLM provider: {str(e)}")

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Simple text generation for multi-query expansion and prompts.
        """
        return self.generate(system_prompt="", user_message=prompt, temperature=temperature)

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate and parse structured JSON output.
        """
        if not self.is_configured or self.client is None:
            raise ValueError("LLM API key is not configured.")

        try:
            config = genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw = response.text.strip() if response.text else "{}"
            return json.loads(raw)
        except Exception as e:
            logger.error(f"LLM JSON generation failed: {e}")
            raise RuntimeError(f"Error generating structured output: {str(e)}")
