from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .models import EvaluationContext, RetrievalResult


class RetrieverEvaluator(ABC):
    """Interface for evaluating retriever quality."""

    @abstractmethod
    def evaluate(self, *, query: str, results: Sequence[RetrievalResult]) -> dict:
        ...


class GeneratorEvaluator(ABC):
    """Interface for evaluating generated answers (e.g., via RAGAS)."""

    @abstractmethod
    def evaluate(self, *, prompt: str, completion: str) -> dict:
        ...


class CompositeEvaluationHook:
    """Combine retriever and generator evaluators when available."""

    def __init__(
        self,
        retriever_evaluators: Sequence[RetrieverEvaluator] | None = None,
        generator_evaluators: Sequence[GeneratorEvaluator] | None = None,
    ) -> None:
        self.retriever_evaluators = list(retriever_evaluators or [])
        self.generator_evaluators = list(generator_evaluators or [])

    def run_retriever_evals(self, query: str, results: Sequence[RetrievalResult]) -> dict:
        report: dict = {}
        for evaluator in self.retriever_evaluators:
            report.update(evaluator.evaluate(query=query, results=results))
        return report

    def run_generator_evals(self, prompt: str, completion: str) -> dict:
        report: dict = {}
        for evaluator in self.generator_evaluators:
            report.update(evaluator.evaluate(prompt=prompt, completion=completion))
        return report

    def summarize(self, context: EvaluationContext) -> EvaluationContext:
        return context
