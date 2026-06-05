"""
Dataset Loader Module
=====================
Load and preprocess benchmark datasets:
  - WebQSP (Yih et al., 2016): Freebase, single-hop and multi-hop questions
  - CWQ (Talmor & Berant, 2018): Freebase, compositional questions
  - GrailQA (Gu et al., 2021): Freebase, generalization to unseen KG elements
  - EntityQuestions (Saxena et al., 2022): Wikidata, direct-fact questions

Supports two modes:
  1. HuggingFace datasets (preferred)
  2. Local JSON/JSONL files (fallback)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset as hf_load_dataset
from loguru import logger


@dataclass
class QASample:
    """A single QA sample from a knowledge graph QA dataset."""
    id: str                         # Unique sample ID
    question: str                   # Natural language question
    answer: list[str]               # List of answer entities (names)
    answer_ids: list[str] = field(default_factory=list)  # KG entity IDs
    topic_entities: list[str] = field(default_factory=list)  # Topic entity IDs
    topic_entity_names: list[str] = field(default_factory=list)
    sparql: str = ""                # Gold SPARQL query (if available)
    kg_backend: str = "freebase"    # Which KG to use
    dataset: str = ""               # Source dataset name

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "answer_ids": self.answer_ids,
            "topic_entities": self.topic_entities,
            "topic_entity_names": self.topic_entity_names,
            "sparql": self.sparql,
            "kg_backend": self.kg_backend,
            "dataset": self.dataset,
        }


class DatasetLoader:
    """Unified dataset loader for all benchmark datasets."""

    SUPPORTED_DATASETS = ["webqsp", "cwq", "grailqa", "entityquestions"]

    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load(self, dataset_name: str, split: str = "test", max_samples: Optional[int] = None) -> list[QASample]:
        """
        Load a dataset by name.

        Args:
            dataset_name: One of "webqsp", "cwq", "grailqa", "entityquestions".
            split: "train", "validation", or "test".
            max_samples: Maximum number of samples to load (None = all).

        Returns:
            List of QASample objects.
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {self.SUPPORTED_DATASETS}")

        logger.info(f"Loading {dataset_name} ({split} split)...")
        loader_fn = getattr(self, f"_load_{dataset_name}")
        samples = loader_fn(split)

        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples

    def _load_webqsp(self, split: str) -> list[QASample]:
        """
        Load WebQSP dataset.
        Original: https://github.com/YihSun/WebQSP
        HuggingFace: rmanluo/RoG-webqsp
        """
        try:
            ds = hf_load_dataset("rmanluo/RoG-webqsp", split=split)
        except Exception:
            return self._load_local("webqsp", split)

        samples = []
        for i, item in enumerate(ds):
            # Parse answers
            answers = item.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]
            elif isinstance(answers, list) and len(answers) > 0 and isinstance(answers[0], dict):
                answers = [a.get("text", a.get("entity_name", str(a))) for a in answers]

            # Parse topic entities
            topic_ents = item.get("topic_entity", [])
            topic_names = item.get("topic_entity_name", [])
            if isinstance(topic_ents, str):
                topic_ents = [topic_ents]
            if isinstance(topic_names, str):
                topic_names = [topic_names]

            sample = QASample(
                id=f"webqsp_{i}",
                question=item["question"],
                answer=answers,
                answer_ids=item.get("answer_id", []),
                topic_entities=topic_ents,
                topic_entity_names=topic_names,
                sparql=item.get("sparql_query", ""),
                kg_backend="freebase",
                dataset="webqsp",
            )
            samples.append(sample)
        return samples

    def _load_cwq(self, split: str) -> list[QASample]:
        """
        Load Complex WebQuestions (CWQ) dataset.
        HuggingFace: rmanluo/RoG-cwq
        """
        try:
            ds = hf_load_dataset("rmanluo/RoG-cwq", split=split)
        except Exception:
            return self._load_local("cwq", split)

        samples = []
        for i, item in enumerate(ds):
            answers = item.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]
            elif isinstance(answers, list) and len(answers) > 0 and isinstance(answers[0], dict):
                answers = [a.get("text", a.get("entity_name", str(a))) for a in answers]

            topic_ents = item.get("topic_entity", [])
            topic_names = item.get("topic_entity_name", [])
            if isinstance(topic_ents, str):
                topic_ents = [topic_ents]
            if isinstance(topic_names, str):
                topic_names = [topic_names]

            sample = QASample(
                id=f"cwq_{i}",
                question=item["question"],
                answer=answers,
                answer_ids=item.get("answer_id", []),
                topic_entities=topic_ents,
                topic_entity_names=topic_names,
                sparql=item.get("sparql_query", ""),
                kg_backend="freebase",
                dataset="cwq",
            )
            samples.append(sample)
        return samples

    def _load_grailqa(self, split: str) -> list[QASample]:
        """
        Load GrailQA dataset.
        HuggingFace: Salesforce/grailqa
        """
        try:
            # Try HuggingFace first
            if split == "validation":
                split = "val"
            ds = hf_load_dataset("Salesforce/grailqa", split=split)
        except Exception:
            return self._load_local("grailqa", split)

        samples = []
        for i, item in enumerate(ds):
            answers = item.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]

            sample = QASample(
                id=f"grailqa_{item.get('qid', i)}",
                question=item["question"],
                answer=answers,
                answer_ids=item.get("answer_id", []),
                topic_entities=[item.get("topic_entity", "")] if item.get("topic_entity") else [],
                topic_entity_names=[item.get("topic_entity_name", "")] if item.get("topic_entity_name") else [],
                sparql=item.get("sparql_query", ""),
                kg_backend="freebase",
                dataset="grailqa",
            )
            samples.append(sample)
        return samples

    def _load_entityquestions(self, split: str) -> list[QASample]:
        """
        Load EntityQuestions dataset.
        HuggingFace: dwaraknath/entityquestions
        """
        try:
            ds = hf_load_dataset("dwaraknath/entityquestions", split=split)
        except Exception:
            return self._load_local("entityquestions", split)

        samples = []
        for i, item in enumerate(ds):
            answers = item.get("answers", item.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]

            sample = QASample(
                id=f"eq_{i}",
                question=item["question"],
                answer=answers,
                answer_ids=item.get("answer_id", []),
                topic_entities=[item.get("entity_id", "")] if item.get("entity_id") else [],
                topic_entity_names=[item.get("entity", "")] if item.get("entity") else [],
                kg_backend="wikidata",
                dataset="entityquestions",
            )
            samples.append(sample)
        return samples

    def _load_local(self, dataset_name: str, split: str) -> list[QASample]:
        """Fallback: load from local JSON/JSONL files."""
        filepath = os.path.join(self.data_dir, f"{dataset_name}_{split}.jsonl")
        if not os.path.exists(filepath):
            filepath_json = os.path.join(self.data_dir, f"{dataset_name}_{split}.json")
            if os.path.exists(filepath_json):
                with open(filepath_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Local data file not found: {filepath}\n"
                    f"Please download the dataset or use HuggingFace datasets."
                )
        else:
            data = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))

        kg_backend = "wikidata" if dataset_name == "entityquestions" else "freebase"
        samples = []
        for i, item in enumerate(data):
            sample = QASample(
                id=f"{dataset_name}_{i}",
                question=item["question"],
                answer=item.get("answer", item.get("answers", [])),
                answer_ids=item.get("answer_id", item.get("answer_ids", [])),
                topic_entities=item.get("topic_entities", []),
                topic_entity_names=item.get("topic_entity_names", []),
                sparql=item.get("sparql_query", ""),
                kg_backend=kg_backend,
                dataset=dataset_name,
            )
            if isinstance(sample.answer, str):
                sample.answer = [sample.answer]
            samples.append(sample)
        return samples


def load_dataset(
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    data_dir: str = "./data/raw",
) -> list[QASample]:
    """Convenience function to load a dataset."""
    loader = DatasetLoader(data_dir=data_dir)
    return loader.load(dataset_name, split, max_samples)
