"""
GPT-4o Reasoning Module (§3.6)
================================
Generates answers from retrieved subgraphs using GPT-4o/GPT-4o-mini.

Two input formats (Table 5 ablation):
  - Triplets: (head, relation, tail) format
  - Natural language: sentence-based description

Based on the paper: "We adopt GPT-4o as the reasoning model... the retrieved
subgraph is converted into textual format and fed into the LLM."
"""

import re
from typing import Optional
from dataclasses import dataclass

from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ReasoningConfig
from kg.base import SubGraph
from llm.llm_client import LLMClient


@dataclass
class ReasoningResult:
    """Result from the reasoning module."""
    answer: str                     # Generated answer text
    answer_entities: list[str]      # Extracted answer entity names
    confidence: float               # Confidence score (0-1)
    reasoning_chain: str            # The reasoning explanation
    raw_response: str               # Raw LLM response


class GPTReasoner:
    """
    GPT-4o reasoning on retrieved subgraphs.

    Takes a subgraph, converts it to text, and asks the LLM to answer
    the question based on the graph evidence.
    """

    def __init__(self, llm: LLMClient, config: Optional[ReasoningConfig] = None):
        self.llm = llm
        self.config = config or ReasoningConfig()

    def reason(
        self,
        question: str,
        subgraph: SubGraph,
        format: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Generate an answer from the question and retrieved subgraph.

        Args:
            question: The natural language question.
            subgraph: The retrieved knowledge graph subgraph.
            format: Output format for subgraph ("triplets" or "natural_language").

        Returns:
            ReasoningResult with the generated answer.
        """
        format = format or self.config.use_graph_format

        # Convert subgraph to text
        graph_text = subgraph.to_text(format=format)

        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(question, graph_text)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph question answering assistant. "
                    "Based on the provided knowledge graph triples, answer the question. "
                    "Provide a concise answer and show your reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(
            messages,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.reasoning_max_tokens,
            model=self.config.reasoning_model,
        )

        return self._parse_response(response, question)

    def _build_reasoning_prompt(self, question: str, graph_text: str) -> str:
        """Build the reasoning prompt with question and graph context."""
        return """Answer the following question based on the provided knowledge graph information.

Question: {question}

Knowledge Graph Triples:
{graph_text}

Instructions:
1. Analyze the knowledge graph triples to find relevant information
2. Reason step by step to derive the answer
3. Provide a concise final answer

Output format:
Reasoning: <your step-by-step reasoning>
Answer: <concise answer, entity name(s) only>
Confidence: <0.0 to 1.0>""".format(
            question=question,
            graph_text=graph_text,
        )

    def _parse_response(self, response: str, question: str) -> ReasoningResult:
        """Parse the LLM response into structured output."""
        # Extract reasoning
        reasoning = ""
        reasoning_match = re.search(
            r"Reasoning:\s*(.*?)(?=Answer:|$)", response, re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract answer
        answer = ""
        answer_match = re.search(r"Answer:\s*(.*?)(?=Confidence:|$)", response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        if not answer:
            # Fallback: use the last line as answer
            lines = response.strip().split("\n")
            answer = lines[-1].strip() if lines else response.strip()

        # Extract confidence
        confidence = 0.5
        conf_match = re.search(r"Confidence:\s*([\d.]+)", response)
        if conf_match:
            confidence = min(1.0, max(0.0, float(conf_match.group(1))))

        # Extract answer entities (comma-separated names)
        answer_entities = [
            e.strip() for e in re.split(r"[,;]", answer) if e.strip()
        ]

        return ReasoningResult(
            answer=answer,
            answer_entities=answer_entities,
            confidence=confidence,
            reasoning_chain=reasoning,
            raw_response=response,
        )

    def reason_batch(
        self,
        questions: list[str],
        subgraphs: list[SubGraph],
    ) -> list[ReasoningResult]:
        """
        Batch reasoning for multiple questions.

        Args:
            questions: List of questions.
            subgraphs: List of subgraphs (same order as questions).

        Returns:
            List of ReasoningResult.
        """
        results = []
        for q, sg in zip(questions, subgraphs):
            try:
                result = self.reason(q, sg)
                results.append(result)
            except Exception as e:
                logger.error(f"Reasoning failed for question: {q[:50]}... Error: {e}")
                results.append(ReasoningResult(
                    answer="",
                    answer_entities=[],
                    confidence=0.0,
                    reasoning_chain="",
                    raw_response="",
                ))
        return results
