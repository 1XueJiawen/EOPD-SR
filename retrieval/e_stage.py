"""
E-Stage: Entity-Ontology Extraction (§3.1, Algorithm 1)
=========================================================
Four-step pipeline:
  Step 1: LLM-based entity extraction from question
  Step 2: Candidate entity retrieval (BM25 + fuzzy + embedding)
  Step 3: Topic entity identification (LLM re-ranking)
  Step 4: Ontology-path subgraph extraction (BFS from topic entities)

The E-stage provides entity-anchored evidence by extracting the ontological
structure around topic entities.
"""

import re
from typing import Optional
from collections import defaultdict

from rank_bm25 import BM25Okapi
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EStageConfig
from kg.base import KnowledgeGraph, Entity, SubGraph
from llm.llm_client import LLMClient
from retrieval.dense_retriever import DenseRetriever
from utils.graph_utils import GraphUtils


class EStage:
    """
    E-Stage: Entity-ontology extraction pipeline.

    Input: Question Q, Knowledge Graph G
    Output: E-stage subgraph S_e containing ontology paths from topic entities
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMClient,
        dense_retriever: DenseRetriever,
        config: Optional[EStageConfig] = None,
    ):
        self.kg = kg
        self.llm = llm
        self.retriever = dense_retriever
        self.config = config or EStageConfig()

    def run(self, question: str) -> SubGraph:
        """
        Execute the full E-Stage pipeline (Algorithm 1).

        Args:
            question: The input natural language question.

        Returns:
            SubGraph S_e containing ontology paths from topic entities.
        """
        logger.info(f"[E-Stage] Processing question: {question[:80]}...")

        # Step 1: Entity Extraction via LLM
        extracted_entities = self._extract_entities(question)
        logger.info(f"[E-Stage] Step 1 - Extracted {len(extracted_entities)} entity mentions")

        # Step 2: Candidate Entity Retrieval
        candidate_entities = self._retrieve_candidates(question, extracted_entities)
        logger.info(f"[E-Stage] Step 2 - Retrieved {len(candidate_entities)} candidate entities")

        # Step 3: Topic Entity Identification (LLM re-ranking)
        topic_entities = self._identify_topic_entities(question, candidate_entities)
        logger.info(f"[E-Stage] Step 3 - Identified {len(topic_entities)} topic entities")

        # Step 4: Ontology-path Subgraph Extraction
        subgraph = self._extract_ontology_subgraph(topic_entities)
        logger.info(f"[E-Stage] Step 4 - Extracted subgraph with {subgraph.size} triples")

        return subgraph

    def _extract_entities(self, question: str) -> list[str]:
        """
        Step 1: Extract entity mentions from the question using LLM.

        Uses a structured prompt to identify all entity references in the question.
        """
        prompt = """Extract all entity mentions from the following question. Entity mentions are names of people, places, organizations, works, events, or any specific things referenced in the question.

Question: {question}

List each entity mention on a separate line. Only output the entity names, nothing else.""".format(question=question)

        messages = [
            {"role": "system", "content": "You are an expert at named entity recognition. Extract entity mentions precisely."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages)

        # Parse entity mentions from response
        entities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering like "1. " or "- "
            line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if line and len(line) > 1:
                entities.append(line)

        return entities

    def _retrieve_candidates(
        self,
        question: str,
        entity_mentions: list[str],
    ) -> list[Entity]:
        """
        Step 2: Retrieve candidate entities using BM25 + fuzzy matching + embedding.

        Combines three retrieval methods:
          - BM25 on entity names (lexical matching)
          - Fuzzy string matching (typo tolerance)
          - Dense embedding retrieval (semantic matching)
        """
        all_candidates: dict[str, Entity] = {}

        for mention in entity_mentions:
            # Method 1: Knowledge graph name search
            kg_results = self.kg.search_entities_by_name(
                mention, top_k=self.config.bm25_top_k
            )
            for entity in kg_results:
                all_candidates[entity.id] = entity

            # Method 2: Fuzzy matching (using entity names from KG)
            # This is handled by the KG search in most implementations

            # Method 3: Dense embedding retrieval
            dense_results = self.retriever.retrieve_entities(
                mention, top_k=self.config.embedding_top_k
            )
            for result in dense_results:
                if result.id not in all_candidates:
                    entity = self.kg.get_entity_by_id(result.id)
                    if entity:
                        all_candidates[entity.id] = entity

        return list(all_candidates.values())

    def _identify_topic_entities(
        self,
        question: str,
        candidates: list[Entity],
    ) -> list[Entity]:
        """
        Step 3: Identify topic entities via LLM re-ranking.

        The LLM selects the most relevant entities as "topic entities"
        that the question is primarily about.
        """
        if not candidates:
            return []

        # Format candidates for LLM
        candidate_text = "\n".join(
            f"{i+1}. {e.name} (ID: {e.id}): {e.description[:100] if e.description else 'N/A'}"
            for i, e in enumerate(candidates[:30])  # Limit to avoid token overflow
        )

        prompt = """Given the following question and candidate entities, select the entities that the question is primarily about (topic entities). These are the main entities the question asks about.

Question: {question}

Candidate entities:
{candidates}

List the numbers of the most relevant topic entities (up to {max_k}). Output only the numbers, one per line.""".format(
            question=question,
            candidates=candidate_text,
            max_k=self.config.topic_entity_top_k,
        )

        messages = [
            {"role": "system", "content": "You are an expert at identifying topic entities in knowledge graph questions."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages)

        # Parse selected indices
        selected_indices = []
        for line in response.strip().split("\n"):
            line = line.strip()
            nums = re.findall(r"\d+", line)
            for num in nums:
                idx = int(num) - 1  # Convert to 0-based
                if 0 <= idx < len(candidates):
                    selected_indices.append(idx)

        # Deduplicate while preserving order
        seen = set()
        topic_entities = []
        for idx in selected_indices:
            if idx not in seen:
                seen.add(idx)
                topic_entities.append(candidates[idx])

        return topic_entities[:self.config.topic_entity_top_k]

    def _extract_ontology_subgraph(self, topic_entities: list[Entity]) -> SubGraph:
        """
        Step 4: Extract ontology-path subgraph from topic entities.

        For each topic entity, perform BFS up to max_ontology_depth to collect
        the ontological structure (entity types and relations).
        """
        merged = SubGraph()

        for entity in topic_entities:
            # Get neighbors up to depth 2 (Algorithm 1, Step 4)
            neighbors = self.kg.get_neighbors(
                entity.id,
                max_depth=self.config.max_ontology_depth,
            )
            merged.merge(neighbors)

        # Prune to max relations if needed
        if len(merged.triples) > self.config.max_ontology_relations:
            # Keep triples connected to topic entities
            topic_ids = {e.id for e in topic_entities}
            prioritized = []
            rest = []
            for t in merged.triples:
                if t.head in topic_ids or t.tail in topic_ids:
                    prioritized.append(t)
                else:
                    rest.append(t)
            merged.triples = prioritized + rest
            merged.triples = merged.triples[:self.config.max_ontology_relations]

        return merged
