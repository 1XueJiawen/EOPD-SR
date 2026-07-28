"""
Freebase Knowledge Graph Backend
=================================
SPARQL-based implementation for Freebase access via local Virtuoso server.
Freebase uses MIDs (Machine IDs) like /m/0284d and relations like type.object.name.

Setup:
  1. Install Virtuoso Open-Source Edition
  2. Load Freebase data dump into Virtuoso
  3. Start Virtuoso on localhost:8890
"""

import re
from typing import Optional

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KGConfig
from kg.base import KnowledgeGraph, Entity, KGTriple, SubGraph


# Freebase namespace prefixes
FB_NS = "http://rdf.freebase.com/ns/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

# SPARQL prefix declarations
PREFIXES = f"""
PREFIX fb: <{FB_NS}>
PREFIX rdf: <{RDF_NS}>
PREFIX rdfs: <{RDFS_NS}>
"""


class FreebaseKG(KnowledgeGraph):
    """Freebase knowledge graph via SPARQL endpoint."""

    def __init__(self, config: Optional[KGConfig] = None):
        self.config = config or KGConfig()
        self.sparql = SPARQLWrapper(self.config.freebase_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(30)

        # Caches
        self._entity_cache: dict[str, Entity] = {}
        self._name_cache: dict[str, str] = {}
        logger.info(f"FreebaseKG initialized: {self.config.freebase_endpoint}")

    def _full_uri(self, mid: str) -> str:
        """Convert MID to full URI."""
        if mid.startswith("http"):
            return mid
        return f"{FB_NS}{mid.lstrip('/')}"

    def _short_mid(self, uri: str) -> str:
        """Extract short MID from full URI."""
        if FB_NS in uri:
            return uri.replace(FB_NS, "/")
        return uri

    def _clean_literal(self, value: str) -> str:
        """Clean SPARQL literal value."""
        value = value.strip('"').strip("'")
        # Remove language tags like @en
        if "@" in value and value.rsplit("@", 1)[-1].isalpha():
            value = value.rsplit("@", 1)[0]
        return value

    def execute_sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL query against Freebase endpoint."""
        full_query = PREFIXES + query
        self.sparql.setQuery(full_query)
        try:
            results = self.sparql.query().convert()
            if "results" in results:
                return results["results"]["bindings"]
            return []
        except SPARQLExceptions.QueryBadFormed as e:
            logger.error(f"Bad SPARQL query: {e}")
            return []
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []

    def _resolve_name(self, entity_id: str) -> str:
        """Resolve entity name from MID."""
        if entity_id in self._name_cache:
            return self._name_cache[entity_id]

        uri = self._full_uri(entity_id)
        query = f"""
        SELECT ?name WHERE {{
            <{uri}> fb:type.object.name ?name .
            FILTER(LANG(?name) = "en" || LANG(?name) = "")
        }} LIMIT 1
        """
        results = self.execute_sparql(query)
        if results:
            name = self._clean_literal(results[0]["name"]["value"])
        else:
            name = entity_id  # Fallback to MID

        self._name_cache[entity_id] = name
        return name

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Look up an entity by its MID."""
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]

        uri = self._full_uri(entity_id)
        query = f"""
        SELECT ?name ?description WHERE {{
            OPTIONAL {{ <{uri}> fb:type.object.name ?name . FILTER(LANG(?name) = "en") }}
            OPTIONAL {{ <{uri}> fb:common.topic.description ?description . FILTER(LANG(?description) = "en") }}
        }} LIMIT 1
        """
        results = self.execute_sparql(query)
        if not results:
            return None

        r = results[0]
        name = self._clean_literal(r["name"]["value"]) if "name" in r else entity_id
        desc = self._clean_literal(r["description"]["value"]) if "description" in r else ""

        entity = Entity(id=entity_id, name=name, description=desc)
        self._entity_cache[entity_id] = entity
        self._name_cache[entity_id] = name
        return entity

    def search_entities_by_name(self, name: str, top_k: int = 10) -> list[Entity]:
        """Search for entities by name using FILTER + CONTAINS."""
        query = f"""
        SELECT ?entity ?name WHERE {{
            ?entity fb:type.object.name ?name .
            FILTER(LANG(?name) = "en")
            FILTER(CONTAINS(LCASE(?name), LCASE("{name}")))
        }} LIMIT {top_k}
        """
        results = self.execute_sparql(query)
        entities = []
        for r in results:
            eid = self._short_mid(r["entity"]["value"])
            ename = self._clean_literal(r["name"]["value"])
            entity = Entity(id=eid, name=ename)
            entities.append(entity)
            self._entity_cache[eid] = entity
            self._name_cache[eid] = ename
        return entities

    def get_relations(self, entity_id: str) -> list[tuple[str, str]]:
        """Get all outgoing relations for an entity."""
        uri = self._full_uri(entity_id)
        query = f"""
        SELECT DISTINCT ?relation ?name WHERE {{
            <{uri}> ?relation ?obj .
            OPTIONAL {{ ?relation rdfs:label ?name . FILTER(LANG(?name) = "en") }}
            FILTER(STRSTARTS(STR(?relation), "{FB_NS}"))
            FILTER(?relation != rdf:type)
        }} LIMIT 200
        """
        results = self.execute_sparql(query)
        relations = []
        for r in results:
            rel_id = self._short_mid(r["relation"]["value"])
            rel_name = self._clean_literal(r["name"]["value"]) if "name" in r else rel_id.split(".")[-1]
            relations.append((rel_id, rel_name))
        return relations

    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
        relation_filter: Optional[list[str]] = None,
    ) -> SubGraph:
        """Get neighbor subgraph via BFS up to max_depth."""
        subgraph = SubGraph()
        visited = {entity_id}
        frontier = [entity_id]

        for depth in range(max_depth):
            next_frontier = []
            for eid in frontier:
                uri = self._full_uri(eid)
                filter_clause = ""
                if relation_filter:
                    rel_uris = " ".join(f"<{self._full_uri(r)}>" for r in relation_filter)
                    filter_clause = f"FILTER(?relation IN ({rel_uris}))"

                query = f"""
                SELECT ?relation ?obj ?rel_name ?obj_name WHERE {{
                    <{uri}> ?relation ?obj .
                    OPTIONAL {{ ?relation rdfs:label ?rel_name . FILTER(LANG(?rel_name) = "en") }}
                    OPTIONAL {{ ?obj fb:type.object.name ?obj_name . FILTER(LANG(?obj_name) = "en") }}
                    FILTER(STRSTARTS(STR(?relation), "{FB_NS}"))
                    FILTER(?relation != rdf:type)
                    FILTER(ISURI(?obj))
                    {filter_clause}
                }} LIMIT 100
                """
                results = self.execute_sparql(query)
                for r in results:
                    rel_id = self._short_mid(r["relation"]["value"])
                    rel_name = self._clean_literal(r["rel_name"]["value"]) if "rel_name" in r else rel_id.split(".")[-1]
                    obj_id = self._short_mid(r["obj"]["value"])
                    obj_name = self._clean_literal(r["obj_name"]["value"]) if "obj_name" in r else obj_id

                    triple = KGTriple(
                        head=eid, head_name=self._resolve_name(eid),
                        relation=rel_id, relation_name=rel_name,
                        tail=obj_id, tail_name=obj_name,
                    )
                    subgraph.add_triple(triple)

                    if obj_id not in visited:
                        visited.add(obj_id)
                        next_frontier.append(obj_id)

            frontier = next_frontier

        return subgraph

    def get_relation_range(self, relation_id: str) -> list[str]:
        """Get example range types for a relation."""
        uri = self._full_uri(relation_id)
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            ?s <{uri}> ?o .
            ?o rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), "{FB_NS}"))
        }} LIMIT 10
        """
        results = self.execute_sparql(query)
        return [self._short_mid(r["type"]["value"]) for r in results]

    def path_exists(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 3,
    ) -> list[list[tuple[str, str, str]]]:
        """Find paths between two entities up to max_length hops."""
        paths = []
        self._dfs_paths(source_id, target_id, max_length, [], set(), paths)
        return paths

    def _dfs_paths(
        self,
        current: str,
        target: str,
        remaining: int,
        current_path: list[tuple[str, str, str]],
        visited: set,
        all_paths: list,
    ):
        """DFS to find paths between entities."""
        if remaining < 0:
            return
        if current == target and current_path:
            all_paths.append(list(current_path))
            return
        if remaining == 0:
            return

        visited.add(current)
        neighbors = self.get_neighbors(current, max_depth=1)
        for triple in neighbors.triples:
            if triple.tail not in visited:
                current_path.append((triple.head_name, triple.relation_name, triple.tail_name))
                self._dfs_paths(triple.tail, target, remaining - 1, current_path, visited, all_paths)
                current_path.pop()
        visited.discard(current)
