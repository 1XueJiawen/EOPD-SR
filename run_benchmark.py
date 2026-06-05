"""
Benchmark Runner
=================
Run EOPD-SR on all benchmark datasets and generate comparison tables.

Usage:
    # Run all benchmarks
    python run_benchmark.py

    # Run specific dataset
    python run_benchmark.py --dataset webqsp --max_samples 100

    # Reproduce Table 1 results
    python run_benchmark.py --table 1

    # Run ablation studies (Tables 5, 6, 7)
    python run_benchmark.py --ablation retrieval_format
"""

import os
import sys
import json
import argparse
import time
from typing import Optional

from tqdm import tqdm
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, Config
from main import EOPDSRPipeline
from evaluation.metrics import compute_metrics, save_results

console = Console()


def run_single_dataset(
    pipeline: EOPDSRPipeline,
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> dict:
    """Run pipeline on a single dataset and return metrics."""
    console.rule(f"[bold blue]Evaluating {dataset_name}")
    start_time = time.time()

    metrics = pipeline.run_benchmark(
        dataset_name,
        split=split,
        max_samples=max_samples,
    )

    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 1)

    return metrics


def run_table1_reproduction(pipeline: EOPDSRPipeline, max_samples: Optional[int] = None):
    """
    Reproduce Table 1: Overall performance on WebQSP and CWQ.

    Table 1 from the paper:
    ┌─────────────────────┬───────────────────────────────────────────┐
    │                     │ WebQSP                        CWQ         │
    │ Method              │ F1    Hits@1  ReCall@50   F1   Hits@1    │
    ├─────────────────────┼───────────────────────────────────────────┤
    │ StructGFM-RoBERTa   │ 72.0  67.0    -          60.6  52.9     │
    │ RoG                 │ 73.3  72.4    92.1       65.1  62.8     │
    │ GNN-RAG             │ 73.4  72.8    94.0       66.4  64.4     │
    │ EPOSR+RoG           │ 75.4  74.8    92.9       69.0  66.7     │
    │ EPOSR+GNN-RAG       │ 76.2  75.8    95.3       70.1  67.6     │
    │ KAPING              │ 73.2  72.7    90.9       66.2  65.3     │
    │ EPOSR+KAPING        │ 75.8  75.0    92.0       69.2  66.4     │
    │ EPOSR+GPT-4o        │ 78.1  77.3    95.0       72.2  70.2     │
    └─────────────────────┴───────────────────────────────────────────┘
    """
    results = {}

    # WebQSP
    results["webqsp"] = run_single_dataset(pipeline, "webqsp", "test", max_samples)

    # CWQ
    results["cwq"] = run_single_dataset(pipeline, "cwq", "test", max_samples)

    # Display results
    table = Table(title="Table 1: Overall Performance (Reproduction)")
    table.add_column("Dataset", style="cyan")
    table.add_column("F1", justify="right")
    table.add_column("Hits@1", justify="right")
    table.add_column("ReCall@50", justify="right")
    table.add_column("Avg Graph Size", justify="right")

    for dataset, metrics in results.items():
        table.add_row(
            dataset,
            f"{metrics.get('F1', 0):.1f}",
            f"{metrics.get('Hits@1', 0):.1f}",
            f"{metrics.get('ReCall@50', 0):.1f}",
            f"{metrics.get('Avg. Graph Size', 0):.0f}",
        )

    console.print(table)
    return results


def run_table2_reproduction(pipeline: EOPDSRPipeline, max_samples: Optional[int] = None):
    """
    Reproduce Table 2: Generalizability on GrailQA and EntityQuestions.
    """
    results = {}

    results["grailqa"] = run_single_dataset(pipeline, "grailqa", "validation", max_samples)
    results["entityquestions"] = run_single_dataset(pipeline, "entityquestions", "test", max_samples)

    table = Table(title="Table 2: Generalizability Results (Reproduction)")
    table.add_column("Dataset", style="cyan")
    table.add_column("F1", justify="right")
    table.add_column("Hits@1", justify="right")

    for dataset, metrics in results.items():
        table.add_row(
            dataset,
            f"{metrics.get('F1', 0):.1f}",
            f"{metrics.get('Hits@1', 0):.1f}",
        )

    console.print(table)
    return results


def run_all_benchmarks(pipeline: EOPDSRPipeline, max_samples: Optional[int] = None):
    """Run all four benchmark datasets."""
    results = {}

    for dataset in ["webqsp", "cwq", "grailqa", "entityquestions"]:
        split = "validation" if dataset == "grailqa" else "test"
        results[dataset] = run_single_dataset(pipeline, dataset, split, max_samples)

    # Display summary table
    table = Table(title="EOPD-SR Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("F1", justify="right")
    table.add_column("Hits@1", justify="right")
    table.add_column("ReCall@50", justify="right")
    table.add_column("Avg Graph Size", justify="right")
    table.add_column("Time (s)", justify="right")

    for dataset, metrics in results.items():
        table.add_row(
            dataset,
            f"{metrics.get('F1', 0):.1f}",
            f"{metrics.get('Hits@1', 0):.1f}",
            f"{metrics.get('ReCall@50', 0):.1f}",
            f"{metrics.get('Avg. Graph Size', 0):.0f}",
            f"{metrics.get('elapsed_seconds', 0):.0f}",
        )

    console.print(table)

    # Save combined results
    save_results(results, "./results/all_benchmarks.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="EOPD-SR Benchmark Runner")
    parser.add_argument("--dataset", type=str, help="Specific dataset to evaluate")
    parser.add_argument("--table", type=int, choices=[1, 2], help="Reproduce specific table")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--max_samples", type=int, help="Max samples per dataset")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--budget", type=int, default=4000, help="Token budget")

    args = parser.parse_args()

    # Setup
    config = get_config()
    config.llm.model = args.model
    config.d_stage.token_budget = args.budget

    pipeline = EOPDSRPipeline(config)

    if args.table == 1:
        run_table1_reproduction(pipeline, args.max_samples)
    elif args.table == 2:
        run_table2_reproduction(pipeline, args.max_samples)
    elif args.dataset:
        run_single_dataset(pipeline, args.dataset, args.split, args.max_samples)
    else:
        run_all_benchmarks(pipeline, args.max_samples)


if __name__ == "__main__":
    main()
