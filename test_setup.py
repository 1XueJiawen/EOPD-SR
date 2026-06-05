"""
Setup Verification Script
==========================
Run this script to verify that all dependencies are installed correctly
and the project structure is properly configured.

Usage:
    python test_setup.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    modules = [
        ("config", "from config import get_config, Config"),
        ("llm", "from llm.llm_client import LLMClient"),
        ("kg.base", "from kg.base import KnowledgeGraph, Entity, KGTriple, SubGraph"),
        ("kg.freebase", "from kg.freebase import FreebaseKG"),
        ("kg.wikidata", "from kg.wikidata import WikidataKG"),
        ("data", "from data.dataloader import load_dataset, QASample"),
        ("retrieval.dense_retriever", "from retrieval.dense_retriever import DenseRetriever"),
        ("retrieval.e_stage", "from retrieval.e_stage import EStage"),
        ("retrieval.p_stage", "from retrieval.p_stage import PStage"),
        ("retrieval.d_stage", "from retrieval.d_stage import DStage"),
        ("reasoning", "from reasoning.gpt_reasoner import GPTReasoner, ReasoningResult"),
        ("evaluation", "from evaluation.metrics import compute_metrics, save_results"),
        ("utils", "from utils.graph_utils import GraphUtils"),
    ]

    all_ok = True
    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"  [OK] {name} module")
        except ImportError as e:
            print(f"  [FAIL] {name} module: {e}")
            all_ok = False

    return all_ok


def test_dependencies():
    """Test that all required packages are installed."""
    print("\nTesting dependencies...")

    required_packages = [
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
        ("SPARQLWrapper", "SPARQLWrapper"),
        ("networkx", "networkx"),
        ("rank_bm25", "rank_bm25"),
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml"),
        ("dotenv", "dotenv"),
        ("tenacity", "tenacity"),
        ("requests", "requests"),
        ("loguru", "loguru"),
        ("rich", "rich"),
    ]

    all_ok = True
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [FAIL] {package_name} - not installed")
            all_ok = False

    return all_ok


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from config import get_config
        config = get_config()
        print(f"  [OK] Config loaded")
        print(f"    - LLM model: {config.llm.model}")
        print(f"    - Embedding model: {config.llm.embedding_model}")
        print(f"    - Token budget: {config.d_stage.token_budget}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EOPD-SR Setup Verification")
    print("=" * 60)

    import_ok = test_imports()
    dep_ok = test_dependencies()
    config_ok = test_config()

    print("\n" + "=" * 60)
    if import_ok and dep_ok and config_ok:
        print("[OK] All tests passed! Setup is correct.")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and fill in your API keys")
        print("  2. Run: python main.py --question 'Who is the president of France?' --kg wikidata")
    else:
        print("[FAIL] Some tests failed. Please check the errors above.")
        if not dep_ok:
            print("\n  Run: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
