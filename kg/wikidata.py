"""
Wikidata Knowledge Graph Backend
==================================
SPARQL-based implementation for Wikidata access via public endpoint.
Wikidata uses QIDs (Q-numbers) for entities and P-numbers for properties.

Reference: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service
"""

import re
from typing import Optional

import requests
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KGConfig
from kg.base import KnowledgeGraph, Entity, KGTriple, SubGraph


# Wikidata namespace
WD_NS = "http://www.wikidata.org/entity/"
WDT_NS = "http://www.wikidata.org/prop/direct/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

PREFIXES = f"""
PREFIX wd: <{WD_NS}>
PREFIX wdt: <{WDT_NS}>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <{RDFS_NS}>
"""


class WikidataKG(KnowledgeGraph):
    """Wikidata knowledge graph via public SPARQL endpoint."""

    def __init__(self, config: Optional[KGConfig] = None):
        self.config = config or KGConfig()
        self.endpoint = self.config.wikidata_endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.wikidata_user_agent,
            "Accept": "application/sparql-results+json",
        })

        self._entity_cache: dict[str, Entity] = {}
        self._name_cache: dict[str, str] = {}
        logger.info(f"WikidataKG initialized: {self.endpoint}")

    def _full_uri(self, qid: str) -> str:
        """Convert QID to full URI."""
        if qid.startswith("http"):
            return qid
        return f"{WD_NS}{qid}"

    def _short_qid(self, uri: str) -> str:
        """Extract short QID from full URI."""
        if WD_NS in uri:
            return uri.replace(WD_NS, "")
        return uri

    def execute_sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL query against Wikidata endpoint."""
        full_query = PREFIXES + query
        try:
            response = self.session.post(
                self.endpoint,
                data={"query": full_query},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()
            if "results" in results:
                return results["results"]["bindings"]
            return []
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []

    def _resolve_name(self, qid: str) -> str:
        """Resolve entity name from QID."""
        if qid in self._name_cache:
            return self._name_cache[qid]

        uri = self._full_uri(qid)
        query = f"""
        SELECT ?name WHERE {{
            <{uri}> rdfs:label ?name .
            FILTER(LANG(?name) = "en")
        }} LIMIT 1
        """
        results = self.execute_sparql(query)
        if results:
            name = results[0]["name"]["value"]
        else:
            name = qid

        self._name_cache[qid] = name
        return name

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Look up an entity by its QID."""
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]

        uri = self._full_uri(entity_id)
        query = f"""
        SELECT ?name ?description WHERE {{
            OPTIONAL {{
                <{uri}> rdfs:label ?name .
                FILTER(LANG(?name) = "en")
            }}
            OPTIONAL {{
                <{uri}> schema:description ?description .
                FILTER(LANG(?description) = "en")
            }}
        }} LIMIT 1
        """
        results = self.execute_sparql(query)
        if not results:
            return None

        r = results[0]
        name = r["name"]["value"] if "name" in r else entity_id
        desc = r["description"]["value"] if "description" in r else ""

        entity = Entity(id=entity_id, name=name, description=desc)
        self._entity_cache[entity_id] = entity
        self._name_cache[entity_id] = name
        return entity

    def search_entities_by_name(self, name: str, top_k: int = 10) -> list[Entity]:
        """Search for entities by name using Wikidata search API."""
        # Use the Wikidata API for entity search (more reliable than SPARQL)
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": top_k,
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            entities = []
            for item in data.get("search", []):
                qid = item["id"]
                ename = item.get("label", qid)
                desc = item.get("description", "")
                entity = Entity(id=qid, name=ename, description=desc)
                entities.append(entity)
                self._entity_cache[qid] = entity
                self._name_cache[qid] = ename
            return entities
        except Exception as e:
            logger.error(f"Wikidata entity search error: {e}")
            return []

    def get_relations(self, entity_id: str) -> list[tuple[str, str]]:
        """Get all outgoing direct relations for an entity."""
        uri = self._full_uri(entity_id)
        query = f"""
        SELECT DISTINCT ?relation ?name WHERE {{
            <{uri}> ?relation ?obj .
            ?property wikibase:directClaim ?relation .
            ?property rdfs:label ?name .
            FILTER(LANG(?name) = "en")
        }} LIMIT 200
        """
        results = self.execute_sparql(query)
        relations = []
        for r in results:
            rel_id = r["relation"]["value"].replace(WDT_NS, "").replace(WD_NS, "")
            rel_name = r["name"]["value"]
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
                    rel_uris = " ".join(f"wdt:{r}" for r in relation_filter)
                    filter_clause = f"FILTER(?relation IN ({rel_uris}))"

                query = f"""
                SELECT ?relation ?obj ?rel_name ?obj_name WHERE {{
                    <{uri}> ?relation ?obj .
                    ?property wikibase:directClaim ?relation .
                    ?property rdfs:label ?rel_name .
                    FILTER(LANG(?rel_name) = "en")
                    OPTIONAL {{
                        ?obj rdfs:label ?obj_name .
                        FILTER(LANG(?obj_name) = "en")
                    }}
                    FILTER(STRSTARTS(STR(?relation), "{WDT_NS}"))
                    {filter_clause}
                }} LIMIT 100
                """
                results = self.execute_sparql(query)
                for r in results:
                    rel_id = r["relation"]["value"].replace(WDT_NS, "")
                    rel_name = r["rel_name"]["value"]
                    obj_id = self._short_qid(r["obj"]["value"])
                    obj_name = r["obj_name"]["value"] if "obj_name" in r else obj_id

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
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            ?s wdt:{relation_id} ?o .
            ?o wdt:P31 ?type .
        }} LIMIT 10
        """
        results = self.execute_sparql(query)
        return [self._short_qid(r["type"]["value"]) for r in results]

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
