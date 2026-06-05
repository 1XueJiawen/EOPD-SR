"""
Graph Utility Functions
========================
Shared utilities for graph manipulation, path operations, and subgraph formatting.
Used across E-Stage, P-Stage, and D-Stage.
"""

import heapq
from collections import defaultdict, deque
from typing import Optional

import networkx as nx
from loguru import logger

from kg.base import KGTriple, SubGraph


class GraphUtils:
    """Static utility methods for graph operations."""

    @staticmethod
    def triples_to_networkx(triples: list[KGTriple]) -> nx.MultiDiGraph:
        """Convert a list of triples to a NetworkX directed graph."""
        G = nx.MultiDiGraph()
        for t in triples:
            G.add_edge(
                t.head, t.tail,
                relation=t.relation,
                relation_name=t.relation_name,
                head_name=t.head_name,
                tail_name=t.tail_name,
            )
        return G

    @staticmethod
    def subgraph_to_networkx(subgraph: SubGraph) -> nx.MultiDiGraph:
        """Convert a SubGraph to a NetworkX directed graph."""
        return GraphUtils.triples_to_networkx(subgraph.triples)

    @staticmethod
    def bfs_subgraph(
        G: nx.MultiDiGraph,
        source: str,
        max_depth: int = 2,
        max_edges: int = 100,
    ) -> list[KGTriple]:
        """
        BFS from source node up to max_depth, collecting triples.
        Returns at most max_edges triples.

        Used in E-Stage (Algorithm 1, Step 4): ontology-path subgraph extraction.
        """
        visited = {source}
        frontier = [source]
        triples = []

        for depth in range(max_depth):
            next_frontier = []
            for node in frontier:
                if len(triples) >= max_edges:
                    return triples
                # Outgoing edges
                for _, neighbor, data in G.out_edges(node, data=True):
                    if len(triples) >= max_edges:
                        return triples
                    triples.append(KGTriple(
                        head=node,
                        head_name=data.get("head_name", node),
                        relation=data.get("relation", ""),
                        relation_name=data.get("relation_name", ""),
                        tail=neighbor,
                        tail_name=data.get("tail_name", neighbor),
                    ))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                # Incoming edges
                for neighbor, _, data in G.in_edges(node, data=True):
                    if len(triples) >= max_edges:
                        return triples
                    triples.append(KGTriple(
                        head=neighbor,
                        head_name=data.get("head_name", neighbor),
                        relation=data.get("relation", "") + "_reverse",
                        relation_name=data.get("relation_name", "") + " [reverse]",
                        tail=node,
                        tail_name=data.get("tail_name", node),
                    ))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        return triples

    @staticmethod
    def find_paths(
        G: nx.MultiDiGraph,
        source: str,
        target: str,
        max_length: int = 3,
        max_paths: int = 50,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Find all simple paths between source and target up to max_length.

        Returns:
            List of paths, each path is a list of (head, relation_name, tail) tuples.
        """
        paths = []
        try:
            for path in nx.all_simple_edge_paths(G, source, target, cutoff=max_length):
                if len(paths) >= max_paths:
                    break
                path_triples = []
                for u, v, key in path:
                    data = G.get_edge_data(u, v, key)
                    path_triples.append((u, data.get("relation_name", ""), v))
                paths.append(path_triples)
        except (nx.NetworkXError, nx.NodeNotFound):
            pass
        return paths

    @staticmethod
    def dijkstra_rerank(
        G: nx.MultiDiGraph,
        source: str,
        targets: list[str],
        relevance_scores: dict[str, float],
        length_penalty: float = 0.1,
        top_k: int = 20,
    ) -> list[tuple[list[str], float]]:
        """
        Dijkstra-based reranking for D-Stage (§3.5).

        Combines path relevance scores with path length penalty.
        Score = Σ node_relevance - λ * path_length

        Args:
            G: The knowledge graph.
            source: Source entity.
            targets: Target entities to reach.
            relevance_scores: Dict of entity_id -> relevance score from LLM.
            length_penalty: λ parameter for path length penalty.
            top_k: Number of top paths to return.

        Returns:
            List of (path, score) tuples sorted by score descending.
        """
        scored_paths = []

        for target in targets:
            try:
                # Use NetworkX shortest path with weighted edges
                path = nx.shortest_path(G, source, target)
                if len(path) - 1 > 3:  # Skip paths longer than 3 hops
                    continue

                # Calculate path score
                path_score = sum(
                    relevance_scores.get(node, 0.0) for node in path
                ) - length_penalty * (len(path) - 1)

                scored_paths.append((path, path_score))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # Sort by score descending
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths[:top_k]

    @staticmethod
    def extract_relation_paths(triples: list[KGTriple]) -> list[str]:
        """
        Extract relation-only paths from a list of triples.
        Used for reasoning path representation.
        """
        return [f"{t.head_name} -> {t.relation_name} -> {t.tail_name}" for t in triples]

    @staticmethod
    def merge_subgraphs(subgraphs: list[SubGraph]) -> SubGraph:
        """Merge multiple subgraphs into one."""
        merged = SubGraph()
        for sg in subgraphs:
            merged.merge(sg)
        return merged

    @staticmethod
    def subgraph_token_count(subgraph: SubGraph, format: str = "triplets") -> int:
        """
        Estimate token count for a subgraph text representation.
        Uses ~1.3 tokens per word as a rough estimate.
        """
        text = subgraph.to_text(format=format)
        word_count = len(text.split())
        return int(word_count * 1.3)

    @staticmethod
    def budget_prune_subgraph(
        subgraph: SubGraph,
        edge_budget: int = 30,
        token_budget: int = 4000,
        entity_relevance: Optional[dict[str, float]] = None,
    ) -> SubGraph:
        """
        Budget-aware subgraph pruning for D-Stage (§3.6).

        Prioritize triples by entity relevance scores, keep top-k within budget.
        """
        if entity_relevance is None:
            entity_relevance = {}

        # Score each triple by average relevance of head and tail
        scored_triples = []
        for t in subgraph.triples:
            head_score = entity_relevance.get(t.head, 0.5)
            tail_score = entity_relevance.get(t.tail, 0.5)
            score = (head_score + tail_score) / 2
            scored_triples.append((score, t))

        # Sort by score descending
        scored_triples.sort(key=lambda x: x[0], reverse=True)

        # Greedily add triples within budget
        pruned = SubGraph()
        for score, triple in scored_triples:
            if len(pruned.triples) >= edge_budget:
                break
            pruned.add_triple(triple)
            if GraphUtils.subgraph_token_count(pruned) > token_budget:
                pruned.triples.pop()
                break

        return pruned

    @staticmethod
    def get_unique_relations(triples: list[KGTriple]) -> list[str]:
        """Get unique relation names from triples."""
        return list(set(t.relation_name for t in triples))

    @staticmethod
    def get_unique_entities(triples: list[KGTriple]) -> list[str]:
        """Get unique entity names from triples."""
        entities = set()
        for t in triples:
            entities.add(t.head_name)
            entities.add(t.tail_name)
        return list(entities)
