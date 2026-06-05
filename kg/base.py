"""
Abstract Knowledge Graph Interface
====================================
Defines the base class for knowledge graph backends (Freebase, Wikidata).
All KG implementations must provide:
  - Entity lookup and name resolution
  - Neighbor retrieval (one-hop and multi-hop)
  - Relation enumeration
  - Subgraph extraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KGTriple:
    """A single (head, relation, tail) triple in the knowledge graph."""
    head: str                   # Entity MID / QID
    head_name: str              # Human-readable name
    relation: str               # Relation PID
    relation_name: str          # Human-readable relation name
    tail: str                   # Entity MID / QID
    tail_name: str              # Human-readable name

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.head_name, self.relation_name, self.tail_name)

    def to_natural_language(self) -> str:
        return f"{self.head_name} --[{self.relation_name}]--> {self.tail_name}"

    def __hash__(self):
        return hash((self.head, self.relation, self.tail))

    def __eq__(self, other):
        if not isinstance(other, KGTriple):
            return False
        return (self.head, self.relation, self.tail) == (other.head, other.relation, other.tail)


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str                     # MID / QID
    name: str                   # Human-readable name
    description: str = ""       # Entity description
    types: list[str] = field(default_factory=list)  # Ontological types
    aliases: list[str] = field(default_factory=list) # Alternative names

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class SubGraph:
    """A subgraph consisting of entities and triples."""
    entities: dict[str, Entity] = field(default_factory=dict)  # id -> Entity
    triples: list[KGTriple] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of triples (edges) in the subgraph."""
        return len(self.triples)

    def add_triple(self, triple: KGTriple):
        """Add a triple and its head/tail entities."""
        self.triples.append(triple)
        if triple.head not in self.entities:
            self.entities[triple.head] = Entity(
                id=triple.head, name=triple.head_name
            )
        if triple.tail not in self.entities:
            self.entities[triple.tail] = Entity(
                id=triple.tail, name=triple.tail_name
            )

    def merge(self, other: "SubGraph"):
        """Merge another subgraph into this one."""
        for triple in other.triples:
            if triple not in self.triples:
                self.add_triple(triple)
        for eid, entity in other.entities.items():
            if eid not in self.entities:
                self.entities[eid] = entity

    def to_text(self, format: str = "triplets") -> str:
        """
        Convert subgraph to text representation for LLM reasoning.

        Args:
            format: "triplets" for (h, r, t) format, "natural_language" for sentences.

        Returns:
            Text representation of the subgraph.
        """
        lines = []
        for triple in self.triples:
            if format == "triplets":
                lines.append(f"({triple.head_name}, {triple.relation_name}, {triple.tail_name})")
            else:
                lines.append(triple.to_natural_language())
        return "\n".join(lines)

    def get_neighbors(self, entity_id: str) -> list[tuple[str, str, str]]:
        """Get all (relation, tail_id, tail_name) triples for an entity."""
        neighbors = []
        for t in self.triples:
            if t.head == entity_id:
                neighbors.append((t.relation_name, t.tail, t.tail_name))
            elif t.tail == entity_id:
                neighbors.append((t.relation_name + " [reverse]", t.head, t.head_name))
        return neighbors


class KnowledgeGraph(ABC):
    """Abstract base class for knowledge graph backends."""

    @abstractmethod
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Look up an entity by its ID (MID/QID)."""
        ...

    @abstractmethod
    def search_entities_by_name(self, name: str, top_k: int = 10) -> list[Entity]:
        """Search for entities by name (fuzzy matching)."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
        relation_filter: Optional[list[str]] = None,
    ) -> SubGraph:
        """
        Get the subgraph of neighbors around an entity.

        Args:
            entity_id: The source entity ID.
            max_depth: Maximum BFS depth (1 or 2).
            relation_filter: Only follow these relations (None = all).

        Returns:
            SubGraph of neighbors.
        """
        ...

    @abstractmethod
    def get_relations(self, entity_id: str) -> list[tuple[str, str]]:
        """Get all relations for an entity. Returns [(relation_id, relation_name), ...]."""
        ...

    @abstractmethod
    def get_relation_range(self, relation_id: str) -> list[str]:
        """Get example range (tail types) for a relation."""
        ...

    @abstractmethod
    def path_exists(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 3,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Find paths between two entities.

        Returns:
            List of paths, where each path is a list of (head, relation, tail) triples.
        """
        ...

    @abstractmethod
    def execute_sparql(self, query: str) -> list[dict]:
        """Execute a raw SPARQL query and return results."""
        ...
