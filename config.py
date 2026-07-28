"""
EOPD-SR Configuration Module
=============================
Centralized configuration for all hyperparameters, API settings, and paths.
Follows Table 4 from the paper for optimal hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env file for API keys


# ─── LLM Configuration ────────────────────────────────────────────────────────
@dataclass
class LLMConfig:
    """LLM API configuration for Azure OpenAI."""
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = "2024-08-01-preview"
    model: str = "gpt-4o-mini"          # Primary LLM (§4.1, ablation: model scaling)
    temperature: float = 0.0             # Greedy decoding for deterministic outputs
    max_tokens: int = 512                # Max output tokens per call
    embedding_model: str = "text-embedding-3-small"  # For bi-encoder dense retrieval
    embedding_dimension: int = 1536      # Embedding dimension

    # Fallback for non-Azure OpenAI
    use_azure: bool = True
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"


# ─── Knowledge Graph Configuration ────────────────────────────────────────────
@dataclass
class KGConfig:
    """Knowledge graph backend configuration."""
    # Freebase (local Virtuoso server, §4.1)
    freebase_endpoint: str = os.getenv(
        "FREEBASE_ENDPOINT", "http://localhost:8890/sparql"
    )
    freebase_graph_uri: str = "http://freebase.org"

    # Wikidata (remote SPARQL endpoint)
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    wikidata_user_agent: str = "EOPD-SR/1.0 (research-paper-reproduction)"

    # Graph namespace prefixes
    freebase_prefix: str = "http://rdf.freebase.com/ns/"
    wikidata_prefix: str = "http://www.wikidata.org/entity/"


# ─── E-Stage Configuration (Entity-Ontology Extraction) ───────────────────────
@dataclass
class EStageConfig:
    """E-Stage: Entity-ontology extraction parameters (Table 4)."""
    # Candidate entity retrieval
    bm25_top_k: int = 50                 # BM25 candidates for entity linking
    embedding_top_k: int = 50            # Dense retrieval candidates
    fuzzy_threshold: int = 80            # Fuzzy string matching score threshold (0-100)

    # Topic entity selection (LLM-based re-ranking)
    topic_entity_top_k: int = 10         # Max topic entities per question

    # Ontology subgraph extraction
    max_ontology_depth: int = 2          # Max BFS depth (Algorithm 1, Step 4)
    max_ontology_relations: int = 100    # Max relations per ontology subgraph


# ─── P-Stage Configuration (Path-Dependency Retrieval) ────────────────────────
@dataclass
class PStageConfig:
    """P-Stage: Path-dependency retrieval parameters (Table 4)."""
    # Relation linking
    relation_bm25_top_k: int = 20        # BM25 candidates for relation linking
    relation_llm_top_k: int = 10         # LLM re-ranked relations to keep
    relation_score_threshold: float = 0.6  # Semantic score threshold

    # Path enumeration
    max_path_length: int = 3             # Max hops in reasoning paths
    max_paths_per_entity: int = 50       # Max paths from each source entity

    # Path pruning (direction dependency)
    direction_consistency_threshold: float = 0.5  # Agreement ratio threshold


# ─── D-Stage Configuration (Dual-View Subgraph Assembly) ──────────────────────
@dataclass
class DStageConfig:
    """D-Stage: Dual-view subgraph assembly parameters (Table 4)."""
    # LLM-guided expansion
    expansion_top_k: int = 10            # Top-k one-hop expansions
    expansion_score_threshold: float = 0.5  # Expansion relevance threshold

    # Dijkstra reranking
    dijkstra_top_k: int = 20             # Top-k paths after Dijkstra reranking
    path_length_penalty: float = 0.1     # λ: penalty for longer paths (§3.5)

    # Budget-aware expansion
    token_budget: int = 4000             # Max tokens for final subgraph (§3.6)
    edge_budget: int = 30                # Max edges in final subgraph


# ─── Reasoning Configuration ──────────────────────────────────────────────────
@dataclass
class ReasoningConfig:
    """Reasoning module configuration (§3.6)."""
    reasoning_model: str = "gpt-4o-mini"  # Can swap to gpt-4o for ablation
    reasoning_temperature: float = 0.0
    reasoning_max_tokens: int = 256
    use_graph_format: str = "triplets"   # "triplets" or "natural_language"
    max_context_length: int = 128000     # Model context window


# ─── Evaluation Configuration ─────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """Evaluation settings (§4.2)."""
    datasets: list = field(default_factory=lambda: [
        "webqsp", "cwq", "grailqa", "entityquestions"
    ])
    batch_size: int = 32                 # Batch size for API calls
    max_workers: int = 4                 # Parallel workers
    save_predictions: bool = True
    output_dir: str = "./results"
    log_dir: str = "./logs"


# ─── Master Configuration ─────────────────────────────────────────────────────
@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    e_stage: EStageConfig = field(default_factory=EStageConfig)
    p_stage: PStageConfig = field(default_factory=PStageConfig)
    d_stage: DStageConfig = field(default_factory=DStageConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global settings
    seed: int = 42
    debug: bool = False
    verbose: bool = True


def get_config() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()

    # Override from environment variables if present
    if os.getenv("EPOSR_DEBUG"):
        config.debug = os.getenv("EPOSR_DEBUG", "false").lower() == "true"
    if os.getenv("EPOSR_SEED"):
        config.seed = int(os.getenv("EPOSR_SEED", "42"))

    return config
