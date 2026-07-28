"""
EOPD-SR Main Pipeline
=======================
End-to-end pipeline for Knowledge Graph Question Answering.

Usage:
    # Run on a single question
    python main.py --question "Who is the president of France?" --kg wikidata

    # Run on a benchmark dataset
    python main.py --dataset webqsp --split test --max_samples 100

    # Run with specific configuration
    python main.py --dataset cwq --model gpt-4o --budget 6000

Architecture:
    Question → E-Stage → P-Stage → D-Stage → Reasoning → Answer
              (Entity)   (Path)    (Merge)    (GPT-4o)
"""

import os
import sys
import json
import argparse
import time
from typing import Optional

from tqdm import tqdm
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, Config
from llm.llm_client import LLMClient
from kg.base import KnowledgeGraph, SubGraph
from kg.freebase import FreebaseKG
from kg.wikidata import WikidataKG
from data.dataloader import load_dataset, QASample
from retrieval.dense_retriever import DenseRetriever
from retrieval.e_stage import EStage
from retrieval.p_stage import PStage
from retrieval.d_stage import DStage
from reasoning.gpt_reasoner import GPTReasoner, ReasoningResult
from evaluation.metrics import compute_metrics, save_results


class EOPDSRPipeline:
    """
    End-to-end EOPD-SR pipeline for KGQA.

    Pipeline: Question → E-Stage → P-Stage → D-Stage → Reasoning → Answer
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        # Initialize components
        self.llm = LLMClient(self.config.llm)
        self.dense_retriever = DenseRetriever(self.llm)

        # KG backends
        self.kg_backends: dict[str, KnowledgeGraph] = {}

        # Pipeline stages (initialized per KG backend)
        self._e_stages: dict[str, EStage] = {}
        self._p_stages: dict[str, PStage] = {}
        self._d_stages: dict[str, DStage] = {}

        # Reasoner
        self.reasoner = GPTReasoner(self.llm, self.config.reasoning)

        logger.info("EOPD-SR Pipeline initialized")

    def _get_kg(self, backend: str) -> KnowledgeGraph:
        """Get or create a KG backend instance."""
        if backend not in self.kg_backends:
            if backend == "freebase":
                self.kg_backends[backend] = FreebaseKG(self.config.kg)
            elif backend == "wikidata":
                self.kg_backends[backend] = WikidataKG(self.config.kg)
            else:
                raise ValueError(f"Unknown KG backend: {backend}")
        return self.kg_backends[backend]

    def _get_stages(self, backend: str) -> tuple[EStage, PStage, DStage]:
        """Get or create pipeline stages for a KG backend."""
        if backend not in self._e_stages:
            kg = self._get_kg(backend)
            self._e_stages[backend] = EStage(kg, self.llm, self.dense_retriever, self.config.e_stage)
            self._p_stages[backend] = PStage(kg, self.llm, self.dense_retriever, self.config.p_stage)
            self._d_stages[backend] = DStage(kg, self.llm, self.config.d_stage)
        return self._e_stages[backend], self._p_stages[backend], self._d_stages[backend]

    def answer_question(
        self,
        question: str,
        kg_backend: str = "freebase",
        topic_entities: Optional[list[str]] = None,
    ) -> tuple[ReasoningResult, SubGraph]:
        """
        Answer a single question through the full EOPD-SR pipeline.

        Args:
            question: The natural language question.
            kg_backend: Which KG to use ("freebase" or "wikidata").
            topic_entities: Optional pre-identified topic entity IDs.

        Returns:
            Tuple of (ReasoningResult, final_subgraph).
        """
        e_stage, p_stage, d_stage = self._get_stages(kg_backend)

        # E-Stage: Entity-ontology extraction
        logger.info("=" * 60)
        logger.info(f"Question: {question}")
        logger.info("=" * 60)

        e_subgraph = e_stage.run(question)

        # Extract topic entities from E-stage results
        if topic_entities:
            kg = self._get_kg(kg_backend)
            topic_ents = [kg.get_entity_by_id(eid) for eid in topic_entities]
            topic_ents = [e for e in topic_ents if e is not None]
        else:
            # Use entities from E-stage subgraph as topic entities
            topic_ents = list(e_subgraph.entities.values())[:self.config.e_stage.topic_entity_top_k]

        # P-Stage: Path-dependency retrieval
        p_subgraph = p_stage.run(question, topic_ents)

        # D-Stage: Dual-view subgraph assembly
        final_subgraph = d_stage.run(question, e_subgraph, p_subgraph, topic_ents)

        # Reasoning
        result = self.reasoner.reason(question, final_subgraph)

        logger.info(f"Answer: {result.answer}")
        logger.info(f"Graph size: {final_subgraph.size} triples")

        return result, final_subgraph

    def run_benchmark(
        self,
        dataset_name: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the pipeline on a benchmark dataset.

        Args:
            dataset_name: Dataset name ("webqsp", "cwq", "grailqa", "entityquestions").
            split: Dataset split ("test", "validation").
            max_samples: Maximum number of samples to evaluate.
            output_dir: Directory to save results.

        Returns:
            Dictionary of evaluation metrics.
        """
        output_dir = output_dir or self.config.eval.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        samples = load_dataset(dataset_name, split, max_samples)

        predictions = []
        gold_answers = []
        graph_sizes = []

        logger.info(f"Running EOPD-SR on {dataset_name} ({len(samples)} samples)...")

        for sample in tqdm(samples, desc=f"Evaluating {dataset_name}"):
            try:
                result, subgraph = self.answer_question(
                    question=sample.question,
                    kg_backend=sample.kg_backend,
                    topic_entities=sample.topic_entities if sample.topic_entities else None,
                )

                predictions.append({
                    "id": sample.id,
                    "question": sample.question,
                    "answer": result.answer,
                    "answer_entities": result.answer_entities,
                    "graph_size": subgraph.size,
                    "confidence": result.confidence,
                })

                gold_answers.append({
                    "id": sample.id,
                    "answer": sample.answer,
                })

                graph_sizes.append(subgraph.size)

            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                predictions.append({
                    "id": sample.id,
                    "question": sample.question,
                    "answer": "",
                    "answer_entities": [],
                    "graph_size": 0,
                })
                gold_answers.append({
                    "id": sample.id,
                    "answer": sample.answer,
                })
                graph_sizes.append(0)

        # Compute metrics
        metrics = compute_metrics(predictions, gold_answers)

        # Save results
        save_path = os.path.join(output_dir, f"{dataset_name}_{split}_results.json")
        save_results({
            "dataset": dataset_name,
            "split": split,
            "metrics": metrics,
            "predictions": predictions,
            "config": {
                "model": self.config.llm.model,
                "e_stage": {
                    "bm25_top_k": self.config.e_stage.bm25_top_k,
                    "embedding_top_k": self.config.e_stage.embedding_top_k,
                    "max_ontology_depth": self.config.e_stage.max_ontology_depth,
                },
                "p_stage": {
                    "relation_llm_top_k": self.config.p_stage.relation_llm_top_k,
                    "max_path_length": self.config.p_stage.max_path_length,
                },
                "d_stage": {
                    "token_budget": self.config.d_stage.token_budget,
                    "edge_budget": self.config.d_stage.edge_budget,
                },
            },
        }, save_path)

        logger.info(f"\n{'='*60}")
        logger.info(f"Results for {dataset_name}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'='*60}")

        return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EOPD-SR: Knowledge Graph QA Pipeline")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--kg", type=str, default="freebase", choices=["freebase", "wikidata"],
                        help="Knowledge graph backend")
    parser.add_argument("--dataset", type=str, help="Benchmark dataset to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--max_samples", type=int, help="Max samples to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--budget", type=int, default=4000, help="Token budget for subgraph")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup config
    config = get_config()
    config.llm.model = args.model
    config.d_stage.token_budget = args.budget
    config.debug = args.debug

    # Setup logging
    if args.debug:
        logger.level("DEBUG")
    else:
        logger.level("INFO")

    # Initialize pipeline
    pipeline = EOPDSRPipeline(config)

    if args.question:
        # Single question mode
        result, subgraph = pipeline.answer_question(args.question, args.kg)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Graph size: {subgraph.size} triples")

    elif args.dataset:
        # Benchmark mode
        metrics = pipeline.run_benchmark(
            args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
        print(f"\nFinal Metrics: {json.dumps(metrics, indent=2)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
