"""Общий интерфейс для всех подходов к поиску.

Ключевая идея бенчмарка: любой подход — это функция
"вопрос -> список кусков текста + сколько это стоило".
Если все пять подходов дают такой ответ, их можно честно сравнить.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragbench.corpus import Chunk

# Отличает «порог не задан, возьми из конфига» от «порог явно отключён».
# Без этого различия нельзя одновременно и настраивать порог в config.py,
# и выключать его в скрипте калибровки, которому нужны сырые оценки.
FROM_CONFIG = object()


@dataclass
class RetrievalResult:
    """Что вернул ретривер на один вопрос."""

    chunks: list[Chunk]                 # найденные куски, по убыванию релевантности
    latency_s: float = 0.0              # сколько заняла стадия поиска
    extra_input_tokens: int = 0         # токены, потраченные ВНУТРИ ретривера
                                        # (у grep-агента он сам зовёт LLM — это не бесплатно)
    extra_output_tokens: int = 0
    trace: list[str] = field(default_factory=list)  # лог действий, для отладки

    # Абсолютный скор лучшего кандидата. Есть только у подходов
    # с cross-encoder: косинус и BM25 дают относительные величины,
    # которые не с чем сравнивать.
    best_score: float | None = None

    # Ретривер считает, что ответа в корпусе нет. Заполняется только
    # когда включён порог отказа. None означает «не знаю, решай сам».
    abstain_hint: bool | None = None

    @property
    def sources(self) -> list[str]:
        """Уникальные файлы-источники, в порядке появления."""
        seen: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen


class BaseRetriever(ABC):
    """Базовый класс. Наследники обязаны реализовать search()."""

    name: str = "base"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def build(self) -> None:
        """Подготовка индекса. Вызывается один раз перед прогоном.

        У grep-подхода тут пусто — в этом половина его прелести.
        """
        return None

    @abstractmethod
    def search(self, question: str, k: int) -> RetrievalResult:
        """Найти k наиболее релевантных чанков."""

    def timed_search(self, question: str, k: int) -> RetrievalResult:
        """Обёртка, которая сама замеряет время. Используйте её в раннере."""
        t0 = time.perf_counter()
        result = self.search(question, k)
        result.latency_s = time.perf_counter() - t0
        return result
