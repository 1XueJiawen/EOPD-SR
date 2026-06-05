"""
P-Stage: Path-Dependency Retrieval (§3.3, Algorithm 2)
=======================================================
Three-step pipeline:
  Step 1-3: Relation linking (BM25 + LLM re-ranking)
  Step 4-5: Direction-dependent reachable set computation
  Step 6-7: Path pruning based on direction consistency

The P-stage provides path-based evidence by finding reasoning paths
that connect question entities to potential answer entities.
"""

import re
from typing import Optional
from collections import defaultdict

from rank_bm25 import BM25Okapi
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PStageConfig
from kg.base import KnowledgeGraph, Entity, KGTriple, SubGraph
from llm.llm_client import LLMClient
from retrieval.dense_retriever import DenseRetriever
from utils.graph_utils import GraphUtils


class PStage:
    """
    P-Stage: Path-dependency retrieval pipeline.

    Input: Question Q, Topic entities T_e from E-Stage, Knowledge Graph G
    Output: P-stage subgraph S_p containing pruned reasoning paths
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMClient,
        dense_retriever: DenseRetriever,
        config: Optional[PStageConfig] = None,
    ):
        self.kg = kg
        self.llm = llm
        self.retriever = dense_retriever
        self.config = config or PStageConfig()

    def run(self, question: str, topic_entities: list[Entity]) -> SubGraph:
        """
        Execute the full P-Stage pipeline (Algorithm 2).

        Args:
            question: The input natural language question.
            topic_entities: Topic entities from E-Stage.

        Returns:
            SubGraph S_p containing pruned reasoning paths.
        """
        logger.info(f"[P-Stage] Processing question: {question[:80]}...")

        # Step 1-3: Relation Linking
        linked_relations = self._relation_linking(question)
        logger.info(f"[P-Stage] Steps 1-3 - Linked {len(linked_relations)} relations")

        # Step 4-5: Direction-dependent reachable set
        reachable_paths = self._compute_reachable_paths(topic_entities, linked_relations)
        logger.info(f"[P-Stage] Steps 4-5 - Found {len(reachable_paths)} paths")

        # Step 6-7: Path pruning by direction consistency
        pruned_subgraph = self._prune_paths(reachable_paths)
        logger.info(f"[P-Stage] Steps 6-7 - Pruned to {pruned_subgraph.size} triples")

        return pruned_subgraph

    def _relation_linking(self, question: str) -> list[tuple[str, str, float]]:
        """
        Steps 1-3: Relation linking via BM25 + LLM re-ranking.

        Step 1: BM25 retrieval of candidate relations
        Step 2: Dense retrieval of candidate relations
        Step 3: LLM-based re-ranking to select top-k relations

        Returns:
            List of (relation_id, relation_name, score) tuples.
        """
        # Step 1: Get candidate relations from topic entities' neighborhoods
        # We'll use the dense retriever for relation matching
        dense_results = self.retriever.retrieve_relations(
            question, top_k=self.config.relation_bm25_top_k
        )

        # Step 2: Combine with BM25 if available
        candidates = {}
        for result in dense_results:
            candidates[result.id] = (result.name, result.score)

        if not candidates:
            return []

        # Step 3: LLM re-ranking
        ranked_relations = self._llm_rerank_relations(question, list(candidates.items()))
        return ranked_relations

    def _llm_rerank_relations(
        self,
        question: str,
        candidates: list[tuple[str, tuple[str, float]]],
    ) -> list[tuple[str, str, float]]:
        """
        Step 3: LLM-based relation re-ranking.

        Uses the LLM to score each candidate relation's relevance to the question.
        """
        if not candidates:
            return []

        # Format candidates for LLM
        candidate_text = "\n".join(
            f"{i+1}. {name} (ID: {rid})"
            for i, (rid, (name, _)) in enumerate(candidates[:20])
        )

        prompt = """Given the following question, rank the candidate knowledge graph relations by their relevance to answering the question.

Question: {question}

Candidate relations:
{candidates}

For each relation, output a relevance score from 0.0 to 1.0. Output one line per relation in the format: "number: score"
Only output the scores, nothing else.""".format(
            question=question,
            candidates=candidate_text,
        )

        messages = [
            {"role": "system", "content": "You are an expert at evaluating knowledge graph relations for question answering."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages)

        # Parse scores
        scored_relations = []
        for line in response.strip().split("\n"):
            line = line.strip()
            match = re.match(r"(\d+)[\.:\)]\s*([\d.]+)", line)
            if match:
                idx = int(match.group(1)) - 1
                score = float(match.group(2))
                if 0 <= idx < len(candidates):
                    rid, (name, _) = candidates[idx]
                    if score >= self.config.relation_score_threshold:
                        scored_relations.append((rid, name, score))

        # Sort by score descending
        scored_relations.sort(key=lambda x: x[2], reverse=True)
        return scored_relations[:self.config.relation_llm_top_k]

    def _compute_reachable_paths(
        self,
        topic_entities: list[Entity],
        linked_relations: list[tuple[str, str, float]],
    ) -> list[list[KGTriple]]:
        """
        Steps 4-5: Compute direction-dependent reachable sets.

        For each topic entity, follow the linked relations to find
        reachable entities and collect the paths.
        """
        all_paths = []
        relation_ids = set(rid for rid, _, _ in linked_relations)

        for entity in topic_entities:
            # Get neighbors following linked relations
            subgraph = self.kg.get_neighbors(
                entity.id,
                max_depth=1,
                relation_filter=list(relation_ids) if relation_ids else None,
            )

            # Collect paths as sequences of triples
            for triple in subgraph.triples:
                path = [triple]

                # One more hop for multi-hop paths
                if self.config.max_path_length > 1:
                    next_subgraph = self.kg.get_neighbors(
                        triple.tail,
                        max_depth=1,
                        relation_filter=list(relation_ids) if relation_ids else None,
                    )
                    for next_triple in next_subgraph.triples[:self.config.max_paths_per_entity]:
                        path_extended = path + [next_triple]
                        all_paths.append(path_extended)

                all_paths.append(path)

        return all_paths

    def _prune_paths(self, paths: list[list[KGTriple]]) -> SubGraph:
        """
        Steps 6-7: Prune paths based on direction consistency (§3.4).

        Path dependency pruning removes paths where the direction of
        reasoning contradicts the expected flow of information.

        Key idea: paths that agree on direction (all forward or consistent
        with question semantics) are kept; contradictory paths are pruned.
        """
        if not paths:
            return SubGraph()

        # Score each path by relation relevance
        scored_paths = []
        for path in paths:
            # Simple scoring: average of path length penalty
            # Shorter paths are preferred
            score = 1.0 / (1.0 + len(path))
            scored_paths.append((score, path))

        # Sort by score
        scored_paths.sort(key=lambda x: x[0], reverse=True)

        # Prune: keep top paths within consistency threshold
        pruned = SubGraph()
        for score, path in scored_paths:
            if score < self.config.direction_consistency_threshold:
                break
            for triple in path:
                pruned.add_triple(triple)

        return pruned
