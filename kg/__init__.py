"""Knowledge graph interface module."""

from .base import KnowledgeGraph
from .freebase import FreebaseKG
from .wikidata import WikidataKG

__all__ = ["KnowledgeGraph", "FreebaseKG", "WikidataKG"]
