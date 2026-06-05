"""
LLM Client Module
==================
Unified interface for LLM calls (GPT-4o-mini) and embedding generation.
Handles Azure OpenAI and standard OpenAI API with retry logic.

Used across all stages:
  - E-Stage: entity extraction, topic entity re-ranking
  - P-Stage: relation re-ranking, path selection
  - D-Stage: relevance scoring, expansion guidance
  - Reasoning: final answer generation
"""

import asyncio
import hashlib
import json
from typing import Optional
from functools import lru_cache

import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLMConfig


class LLMClient:
    """Unified LLM client with caching and retry logic."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._setup_client()
        self._cache: dict[str, str] = {}
        self._embedding_cache: dict[str, list[float]] = {}

    def _setup_client(self):
        """Initialize OpenAI client (Azure or standard)."""
        if self.config.use_azure:
            self.client = openai.AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.api_version,
            )
            logger.info(f"Initialized Azure OpenAI client: {self.config.model}")
        else:
            self.client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
            logger.info(f"Initialized OpenAI client: {self.config.model}")

    def _cache_key(self, messages: list[dict], **kwargs) -> str:
        """Generate cache key from messages and parameters."""
        content = json.dumps(messages, sort_keys=True) + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            model: Override default model.
            use_cache: Whether to use response caching.

        Returns:
            The assistant's response text.
        """
        key = self._cache_key(messages, temperature=temperature, max_tokens=max_tokens)
        if use_cache and key in self._cache:
            return self._cache[key]

        response = self.client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        result = response.choices[0].message.content.strip()

        if use_cache:
            self._cache[key] = result

        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        embedding = response.data[0].embedding
        self._embedding_cache[text] = embedding
        return embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embeddings_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for API calls.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                all_embeddings.append((i, self._embedding_cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch embed uncached texts
        for start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=batch,
            )
            for j, data in enumerate(response.data):
                embedding = data.embedding
                idx = uncached_indices[start + j]
                self._embedding_cache[batch[j]] = embedding
                all_embeddings.append((idx, embedding))

        # Sort by original index
        all_embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in all_embeddings]

    async def achat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Async version of chat for parallel calls."""
        key = self._cache_key(messages, temperature=temperature, max_tokens=max_tokens)
        if key in self._cache:
            return self._cache[key]

        # Use sync client in thread pool for simplicity
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.chat(messages, temperature, max_tokens, model, use_cache=False),
        )
        self._cache[key] = result
        return result

    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        self._embedding_cache.clear()
        logger.info("LLM client caches cleared")

    @property
    def cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "chat_cache_size": len(self._cache),
            "embedding_cache_size": len(self._embedding_cache),
        }
