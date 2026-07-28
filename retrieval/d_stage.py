"""
D-Stage: Dual-View Subgraph Assembly (§3.4-3.5)
==================================================
Four-step pipeline:
  Step 1: LLM-guided one-hop expansion
  Step 2: Dijkstra reranking
  Step 3: Dual-view merge (E-stage + P-stage subgraphs)
  Step 4: Budget-aware expansion and pruning

The D-stage combines entity-ontology evidence (E-stage) with
path-dependency evidence (P-stage) into a unified, compact subgraph
optimized for LLM reasoning.
"""

from typing import Optional
from collections import defaultdict

from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DStageConfig
from kg.base import KnowledgeGraph, Entity, KGTriple, SubGraph
from llm.llm_client import LLMClient
from utils.graph_utils import GraphUtils


class DStage:
    """
    D-Stage: Dual-view subgraph assembly pipeline.

    Input: E-stage subgraph S_e, P-stage subgraph S_p, Topic entities, Knowledge Graph
    Output: Final reasoning subgraph S_r
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMClient,
        config: Optional[DStageConfig] = None,
    ):
        self.kg = kg
        self.llm = llm
        self.config = config or DStageConfig()

    def run(
        self,
        question: str,
        e_stage_subgraph: SubGraph,
        p_stage_subgraph: SubGraph,
        topic_entities: list[Entity],
    ) -> SubGraph:
        """
        Execute the full D-Stage pipeline.

        Args:
            question: The input natural language question.
            e_stage_subgraph: Subgraph from E-Stage.
            p_stage_subgraph: Subgraph from P-Stage.
            topic_entities: Topic entities identified in E-Stage.

        Returns:
            Final reasoning subgraph S_r.
        """
        logger.info(f"[D-Stage] Assembling dual-view subgraph...")

        # Step 1: LLM-guided one-hop expansion
        expanded_subgraph = self._llm_guided_expansion(
            question, e_stage_subgraph, p_stage_subgraph, topic_entities
        )
        logger.info(f"[D-Stage] Step 1 - Expanded subgraph: {expanded_subgraph.size} triples")

        # Step 2: Dijkstra reranking
        reranked_subgraph = self._dijkstra_rerank(
            question, expanded_subgraph, topic_entities
        )
        logger.info(f"[D-Stage] Step 2 - Reranked subgraph: {reranked_subgraph.size} triples")

        # Step 3: Dual-view merge
        merged_subgraph = self._dual_view_merge(
            e_stage_subgraph, p_stage_subgraph, reranked_subgraph
        )
        logger.info(f"[D-Stage] Step 3 - Merged subgraph: {merged_subgraph.size} triples")

        # Step 4: Budget-aware expansion
        final_subgraph = self._budget_aware_prune(question, merged_subgraph)
        logger.info(f"[D-Stage] Step 4 - Final subgraph: {final_subgraph.size} triples")

        return final_subgraph

    def _llm_guided_expansion(
        self,
        question: str,
        e_subgraph: SubGraph,
        p_subgraph: SubGraph,
        topic_entities: list[Entity],
    ) -> SubGraph:
        """
        Step 1: LLM-guided one-hop expansion.

        The LLM identifies which entities in the current subgraph are most
        promising for expansion, then adds their one-hop neighbors.
        """
        # Merge E and P subgraphs for initial context
        combined = SubGraph()
        combined.merge(e_subgraph)
        combined.merge(p_subgraph)

        if not combined.triples:
            return combined

        # Get unique entities in the subgraph
        entity_ids = set()
        for t in combined.triples:
            entity_ids.add(t.head)
            entity_ids.add(t.tail)

        # Ask LLM which entities to expand
        entity_text = "\n".join(
            f"- {combined.entities[eid].name} (ID: {eid})"
            for eid in list(entity_ids)[:30]
            if eid in combined.entities
        )

        prompt = """Given the following question and entities from a knowledge graph subgraph, identify which entities should be expanded (i.e., whose neighbors should be explored) to better answer the question.

Question: {question}

Current entities:
{entities}

List the entity IDs that should be expanded, one per line. Only output IDs, nothing else.""".format(
            question=question,
            entities=entity_text,
        )

        messages = [
            {"role": "system", "content": "You are an expert at knowledge graph reasoning. Select entities for expansion."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages)

        # Parse entity IDs to expand
        expand_ids = set()
        for line in response.strip().split("\n"):
            line = line.strip()
            if line in entity_ids:
                expand_ids.add(line)

        # Expand selected entities
        expanded = SubGraph()
        expanded.merge(combined)

        for eid in expand_ids:
            neighbors = self.kg.get_neighbors(eid, max_depth=1)
            # Only add top-k neighbors
            for triple in neighbors.triples[:self.config.expansion_top_k]:
                expanded.add_triple(triple)

        return expanded

    def _dijkstra_rerank(
        self,
        question: str,
        subgraph: SubGraph,
        topic_entities: list[Entity],
    ) -> SubGraph:
        """
        Step 2: Dijkstra reranking (§3.5).

        Use Dijkstra's algorithm to find shortest paths from topic entities
        to all other entities, weighted by relevance scores.
        """
        if not subgraph.triples or not topic_entities:
            return subgraph

        # Build NetworkX graph
        G = GraphUtils.subgraph_to_networkx(subgraph)

        # Compute entity relevance scores via LLM
        entity_ids = list(subgraph.entities.keys())
        relevance_scores = self._compute_entity_relevance(question, entity_ids)

        # Dijkstra reranking from each topic entity
        topic_ids = {e.id for e in topic_entities}
        target_ids = [eid for eid in entity_ids if eid not in topic_ids]

        reranked_triples = set()
        for topic in topic_entities:
            scored_paths = GraphUtils.dijkstra_rerank(
                G, topic.id, target_ids,
                relevance_scores=relevance_scores,
                length_penalty=self.config.path_length_penalty,
                top_k=self.config.dijkstra_top_k,
            )

            # Collect triples from top paths
            for path, score in scored_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = G.get_edge_data(u, v)
                    if edge_data:
                        for key, data in edge_data.items():
                            reranked_triples.add(KGTriple(
                                head=u,
                                head_name=data.get("head_name", u),
                                relation=data.get("relation", ""),
                                relation_name=data.get("relation_name", ""),
                                tail=v,
                                tail_name=data.get("tail_name", v),
                            ))

        result = SubGraph()
        for triple in reranked_triples:
            result.add_triple(triple)
        return result

    def _compute_entity_relevance(
        self,
        question: str,
        entity_ids: list[str],
    ) -> dict[str, float]:
        """
        Compute relevance scores for entities using LLM.
        """
        if not entity_ids:
            return {}

        # Get entity names
        entity_names = []
        for eid in entity_ids:
            entity = self.kg.get_entity_by_id(eid)
            entity_names.append(entity.name if entity else eid)

        # Batch scoring (groups of 20)
        scores = {}
        batch_size = 20

        for start in range(0, len(entity_ids), batch_size):
            batch_ids = entity_ids[start:start + batch_size]
            batch_names = entity_names[start:start + batch_size]

            entity_text = "\n".join(
                f"{i+1}. {name} (ID: {eid})"
                for i, (eid, name) in enumerate(zip(batch_ids, batch_names))
            )

            prompt = """Score each entity's relevance to answering the following question. Output a score from 0.0 to 1.0 for each.

Question: {question}

Entities:
{entities}

Output one score per line in the format: "number: score"""".format(
                question=question,
                entities=entity_text,
            )

            messages = [
                {"role": "system", "content": "Score entity relevance for knowledge graph QA."},
                {"role": "user", "content": prompt},
            ]

            response = self.llm.chat(messages)

            import re
            for line in response.strip().split("\n"):
                match = re.match(r"(\d+)[\.:\)]\s*([\d.]+)", line)
                if match:
                    idx = int(match.group(1)) - 1
                    score = float(match.group(2))
                    if 0 <= idx < len(batch_ids):
                        scores[batch_ids[idx]] = score

        return scores

    def _dual_view_merge(
        self,
        e_subgraph: SubGraph,
        p_subgraph: SubGraph,
        expanded: SubGraph,
    ) -> SubGraph:
        """
        Step 3: Dual-view merge.

        Combines evidence from all three sources:
          - E-stage: entity-ontology structure
          - P-stage: path-dependency evidence
          - Expansion: LLM-guided additional context
        """
        merged = SubGraph()
        merged.merge(e_subgraph)
        merged.merge(p_subgraph)
        merged.merge(expanded)
        return merged

    def _budget_aware_prune(
        self,
        question: str,
        subgraph: SubGraph,
    ) -> SubGraph:
        """
        Step 4: Budget-aware expansion and pruning (§3.6).

        Ensures the final subgraph fits within the token budget
        for LLM reasoning context.
        """
        # Compute entity relevance for prioritization
        entity_ids = list(subgraph.entities.keys())
        relevance_scores = self._compute_entity_relevance(question, entity_ids)

        # Budget-aware pruning
        pruned = GraphUtils.budget_prune_subgraph(
            subgraph,
            edge_budget=self.config.edge_budget,
            token_budget=self.config.token_budget,
            entity_relevance=relevance_scores,
        )

        return pruned
