"""
FinSight AI - OpenAI Client
=============================
Thin wrapper around the OpenAI API for generating grounded answers.

This module handles:
- Loading API key and model name from environment
- Sending chat completion requests to OpenAI
- Error handling (auth, rate-limit, timeout)

Author: FinSight AI Team
Phase: 2 (Generation Layer)
"""

import os
from typing import Optional

# OpenAI SDK (v1+)
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# OPENAI CLIENT WRAPPER
# =============================================================================

class OpenAIClient:
    """
    Wrapper around the OpenAI API for generating answers.
    
    Why a wrapper?
    - Centralizes API key management
    - Handles errors in one place
    - Makes it easy to swap models later
    - Keeps main.py clean
    
    Usage:
        client = OpenAIClient()
        answer = client.generate("You are a helpful assistant.", "What is 2+2?")
    """
    
    def __init__(self):
        """
        Initialize the OpenAI client.
        
        Reads OPENAI_API_KEY and OPENAI_MODEL from environment.
        Does NOT crash if key is missing — instead, sets a flag so the
        /chat endpoint can return a helpful error.
        """
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Check if API key is configured
        self.is_configured = bool(self.api_key and self.api_key != "your-api-key-here")
        
        if self.is_configured:
            # Initialize the OpenAI client
            self.client = OpenAI(api_key=self.api_key)
            print(f"🔑 OpenAI client initialized (model: {self.model})")
        else:
            self.client = None
            print("⚠️  OpenAI API key not configured.")
            print("   The /chat endpoint will not work until you set OPENAI_API_KEY in .env")
            print("   The /retrieve and /health endpoints will continue to work normally.")
    
    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a response using OpenAI's Chat Completion API.
        
        How it works:
        1. Send a system prompt (with grounding rules + context)
        2. Send the user's question
        3. Get back the model's response
        
        We use temperature=0 for deterministic, grounded answers.
        This minimizes creativity/hallucination — the model sticks
        closely to the provided context.
        
        Args:
            system_prompt: The full system prompt (includes context + grounding rules)
            user_message: The user's question
            
        Returns:
            The model's response text
            
        Raises:
            ValueError: If API key is not configured
            RuntimeError: If API call fails
        """
        # Guard: Check if client is configured
        if not self.is_configured or self.client is None:
            raise ValueError(
                "OpenAI API key is not configured. "
                "Please set OPENAI_API_KEY in your .env file and restart the server."
            )
        
        try:
            # Make the API call
            # temperature=0 → deterministic output (no creativity)
            # This is crucial for grounded answers
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,       # Deterministic — no hallucination
                max_tokens=1000,     # Balanced: structured output without excess latency
            )
            
            # Extract the response text
            answer = response.choices[0].message.content.strip()
            return answer
            
        except AuthenticationError:
            raise RuntimeError(
                "Invalid OpenAI API key. Please check OPENAI_API_KEY in your .env file."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI API rate limit exceeded. Please wait a moment and try again."
            )
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling OpenAI: {str(e)}")
