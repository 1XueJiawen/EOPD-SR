"""
Dense Retriever Module (§3.2)
==============================
Bi-encoder model for candidate entity and relation retrieval.
Uses OpenAI text-embedding-3-small for encoding.

Components:
  - Candidate entity retrieval: encode mention, search over entity name corpus
  - Relation linking: encode question, search over relation name corpus
  - Score computation: cosine similarity between query and candidate embeddings

Based on the paper: "We employ a pre-trained bi-encoder model to compute
the semantic similarity between the question and candidate entities/relations."
"""

import os
import json
import pickle
from typing import Optional
from dataclasses import dataclass

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_client import LLMClient


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    id: str              # Entity or relation ID
    name: str            # Human-readable name
    score: float         # Similarity score


class DenseRetriever:
    """
    Bi-encoder dense retriever using OpenAI embeddings.

    Two main functionalities:
      1. Entity retrieval: given a mention string, find candidate entities
      2. Relation retrieval: given a question, find candidate relations

    Embeddings are cached to disk for efficiency.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache_dir: str = "./cache/embeddings",
    ):
        self.llm = llm_client
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # In-memory embedding indices
        self._entity_names: list[str] = []
        self._entity_ids: list[str] = []
        self._entity_embeddings: Optional[np.ndarray] = None

        self._relation_names: list[str] = []
        self._relation_ids: list[str] = []
        self._relation_embeddings: Optional[np.ndarray] = None

    def build_entity_index(
        self,
        entity_ids: list[str],
        entity_names: list[str],
        batch_size: int = 128,
    ):
        """
        Build dense index for entity candidates.

        Args:
            entity_ids: List of entity IDs (MID/QID).
            entity_names: List of entity names (same order as IDs).
            batch_size: Batch size for embedding API calls.
        """
        self._entity_ids = entity_ids
        self._entity_names = entity_names

        # Try loading cached embeddings
        cache_path = os.path.join(self.cache_dir, "entity_embeddings.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                if cached["ids"] == entity_ids:
                    self._entity_embeddings = cached["embeddings"]
                    logger.info(f"Loaded cached entity embeddings: {len(entity_ids)} entities")
                    return

        logger.info(f"Building entity index for {len(entity_names)} entities...")
        embeddings = self.llm.get_embeddings_batch(entity_names, batch_size=batch_size)
        self._entity_embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self._entity_embeddings, axis=1, keepdims=True)
        self._entity_embeddings = self._entity_embeddings / (norms + 1e-8)

        # Cache
        with open(cache_path, "wb") as f:
            pickle.dump({"ids": entity_ids, "embeddings": self._entity_embeddings}, f)

        logger.info(f"Entity index built: {len(entity_ids)} entities, dim={self._entity_embeddings.shape[1]}")

    def build_relation_index(
        self,
        relation_ids: list[str],
        relation_names: list[str],
        batch_size: int = 128,
    ):
        """
        Build dense index for relation candidates.

        Args:
            relation_ids: List of relation IDs.
            relation_names: List of relation names.
            batch_size: Batch size for embedding API calls.
        """
        self._relation_ids = relation_ids
        self._relation_names = relation_names

        cache_path = os.path.join(self.cache_dir, "relation_embeddings.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                if cached["ids"] == relation_ids:
                    self._relation_embeddings = cached["embeddings"]
                    logger.info(f"Loaded cached relation embeddings: {len(relation_ids)} relations")
                    return

        logger.info(f"Building relation index for {len(relation_names)} relations...")
        embeddings = self.llm.get_embeddings_batch(relation_names, batch_size=batch_size)
        self._relation_embeddings = np.array(embeddings, dtype=np.float32)

        norms = np.linalg.norm(self._relation_embeddings, axis=1, keepdims=True)
        self._relation_embeddings = self._relation_embeddings / (norms + 1e-8)

        with open(cache_path, "wb") as f:
            pickle.dump({"ids": relation_ids, "embeddings": self._relation_embeddings}, f)

        logger.info(f"Relation index built: {len(relation_ids)} relations")

    def retrieve_entities(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[RetrievalResult]:
        """
        Retrieve candidate entities for a query mention.

        Args:
            query: The entity mention or description.
            top_k: Number of top candidates to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if self._entity_embeddings is None:
            logger.warning("Entity index not built. Call build_entity_index first.")
            return []

        query_emb = np.array(self.llm.get_embedding(query), dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Cosine similarity (dot product since vectors are normalized)
        scores = self._entity_embeddings @ query_emb

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                id=self._entity_ids[idx],
                name=self._entity_names[idx],
                score=float(scores[idx]),
            ))
        return results

    def retrieve_relations(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve candidate relations for a question.

        Args:
            query: The question text.
            top_k: Number of top candidates to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if self._relation_embeddings is None:
            logger.warning("Relation index not built. Call build_relation_index first.")
            return []

        query_emb = np.array(self.llm.get_embedding(query), dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        scores = self._relation_embeddings @ query_emb

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                id=self._relation_ids[idx],
                name=self._relation_names[idx],
                score=float(scores[idx]),
            ))
        return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = np.array(self.llm.get_embedding(text1), dtype=np.float32)
        emb2 = np.array(self.llm.get_embedding(text2), dtype=np.float32)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
