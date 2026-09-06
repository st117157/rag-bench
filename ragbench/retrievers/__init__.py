"""Реестр ретриверов.

Весь смысл проекта в том, что все пять подходов реализуют
один и тот же интерфейс (см. base.py) и потому сравнимы честно.
Добавили новый подход — зарегистрировали здесь, и он автоматически
попадает в бенчмарк.
"""

from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.retrievers.grep_agent import GrepAgentRetriever
from ragbench.retrievers.hybrid import HybridRetriever
from ragbench.retrievers.hybrid_gated import HybridGatedRetriever
from ragbench.retrievers.longctx import (
    LongContextFilteredRetriever,
    LongContextRetriever,
)
from ragbench.retrievers.naive import NaiveDenseRetriever
from ragbench.retrievers.reranked import RerankedRetriever

# Подходы, которым для работы нужна LLM. В режиме --retrieval-only
# они пропускаются.
NEEDS_LLM = {"grep"}

REGISTRY: dict[str, type[BaseRetriever]] = {
    "naive": NaiveDenseRetriever,
    "hybrid": HybridRetriever,
    "reranked": RerankedRetriever,
    "hybrid_gated": HybridGatedRetriever,
    "grep": GrepAgentRetriever,
    "longctx": LongContextRetriever,
    "longctx_filtered": LongContextFilteredRetriever,
}

__all__ = ["NEEDS_LLM", "REGISTRY", "BaseRetriever", "RetrievalResult"]
