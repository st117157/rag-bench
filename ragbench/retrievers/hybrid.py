"""Подход 2: гибридный поиск. BM25 (слова) + эмбеддинги (смысл).

Зачем: у двух методов разные слепые зоны.
  - BM25 не понимает синонимов: «вайфай не пашет» != «сбой подключения».
  - Эмбеддинги плохо ловят редкие точные термины, имена функций,
    коды ошибок, номера версий.
Вместе они закрывают дыры друг друга, и на практике гибрид почти
всегда обходит чистые вектора при минимальной цене реализации.

Объединяем два списка через RRF (Reciprocal Rank Fusion):
    score(doc) = sum по методам от  вес_метода / (K + место_в_этом_методе)
Хорош тем, что не требует приводить к общей шкале несравнимые
скоры BM25 и косинуса — работает только с местами в рейтинге.

Вес задаётся HYBRID_ALPHA в config.py: alpha идёт векторам,
(1 - alpha) — словам. При alpha = 0.5 это классический RRF.
"""

from __future__ import annotations

import re

from config import CANDIDATES_K, HYBRID_ALPHA
from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.retrievers.naive import NaiveDenseRetriever

RRF_K = 60  # стандартная константа из статьи про RRF; менять почти не нужно


def tokenize(text: str) -> list[str]:
    """Простейшая токенизация для BM25.

    Здесь теряется часть сигнала. Что можно улучшить: выкинуть стоп-слова
    и — существенно для документации по коду — не разбивать идентификаторы
    вида `response_model` на два токена.
    """
    return re.findall(r"[a-zA-Zа-яА-Я_][a-zA-Z0-9а-яА-Я_]+", text.lower())


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, chunks, alpha: float | None = None) -> None:
        super().__init__(chunks)
        self.alpha = HYBRID_ALPHA if alpha is None else alpha
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("HYBRID_ALPHA должен быть в диапазоне от 0 до 1")
        self.dense = NaiveDenseRetriever(chunks)
        self.bm25 = None
        # id -> позиция в списке. Нужно, чтобы переводить результаты
        # dense-поиска обратно в индексы без медленного list.index().
        self._pos = {c.id: i for i, c in enumerate(chunks)}

    def build(self) -> None:
        from rank_bm25 import BM25Okapi

        self.dense.build()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def _bm25_ranking(self, question: str, n: int) -> list[int]:
        """Вернуть индексы n лучших чанков по BM25."""
        scores = self.bm25.get_scores(tokenize(question))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    def search(self, question: str, k: int) -> RetrievalResult:
        n = max(CANDIDATES_K, k)

        dense_ids = [self._pos[c.id] for c in self.dense.search(question, n).chunks]
        bm25_ids = self._bm25_ranking(question, n)

        fused: dict[int, float] = {}
        for ranking, weight in ((dense_ids, self.alpha), (bm25_ids, 1 - self.alpha)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)

        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]

        # Пересечение топов двух методов — полезная диагностика.
        # Близко к k: методы находят одно и то же, гибрид ничего не даёт.
        # Близко к нулю: методы смотрят в разные стороны, и именно здесь
        # гибрид выигрывает больше всего.
        overlap = len(set(dense_ids[:k]) & set(bm25_ids[:k]))
        return RetrievalResult(
            chunks=[self.chunks[i] for i in best],
            trace=[
                f"RRF alpha={self.alpha}: пересечение топ-{k} двух методов = {overlap}"
            ],
        )
