"""Подход 3: гибрид + cross-encoder реранкер.

Разница между эмбеддингами и cross-encoder'ом:
  - Эмбеддинг кодирует вопрос и документ ПОРОЗНЬ, потом сравнивает
    два вектора. Быстро (индекс считается заранее), но грубо.
  - Cross-encoder подаёт пару (вопрос, документ) в модель ВМЕСТЕ
    и выдаёт оценку релевантности. Точно, но медленно —
    заранее ничего не посчитать, каждая пара считается в момент запроса.

Отсюда стандартная двухэтапная схема:
  этап 1 (дёшево) — гибридом достаём CANDIDATES_K кандидатов,
  этап 2 (дорого) — реранкером пересортировываем их и берём топ-k.

ВТОРАЯ, НЕДООЦЕНЁННАЯ РОЛЬ РЕРАНКЕРА

Косинусное расстояние относительно: 0.62 само по себе не значит ничего,
это просто «ближе, чем 0.51». Поэтому по нему нельзя понять, есть ли
вообще ответ в корпусе — ближайшие соседи существуют всегда, даже когда
в базе нет ничего подходящего.

Cross-encoder даёт АБСОЛЮТНУЮ оценку релевантности пары. Её уже можно
сравнить с порогом: лучший кандидат ниже порога — ответа нет,
и правильное поведение системы здесь молчать, а не сочинять.

Это и есть механизм честного «я не знаю». Порог задаётся в config.py
в словаре ABSTAIN_THRESHOLDS под ключом "reranked" и подбирается скриптом:
    python scripts/05_analyze_abstention.py --method reranked
"""

from __future__ import annotations

from config import ABSTAIN_THRESHOLDS, CANDIDATES_K, RERANKER_MODEL
from ragbench.retrievers.base import FROM_CONFIG, RetrievalResult
from ragbench.retrievers.hybrid import HybridRetriever


class RerankedRetriever(HybridRetriever):
    name = "reranked"

    def __init__(self, chunks, alpha: float | None = None,
                 threshold=FROM_CONFIG) -> None:
        super().__init__(chunks, alpha)
        self.reranker = None
        # threshold не передан  -> берём из конфига по имени подхода
        # threshold = None      -> механизм явно выключен (нужно калибровке)
        # threshold = число     -> используем его
        self.threshold = (
            ABSTAIN_THRESHOLDS.get(self.name)
            if threshold is FROM_CONFIG
            else threshold
        )

    def build(self) -> None:
        from sentence_transformers import CrossEncoder

        super().build()
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def search(self, question: str, k: int) -> RetrievalResult:
        # Этап 1: много дешёвых кандидатов.
        candidates = super().search(question, CANDIDATES_K).chunks
        if not candidates:
            return RetrievalResult(chunks=[], abstain_hint=True)

        # Этап 2: дорогая, но точная пересортировка.
        pairs = [(question, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)

        ordered = sorted(
            zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True
        )
        top = [c for c, _ in ordered[:k]]
        best = float(ordered[0][1])

        moved = sum(1 for i, c in enumerate(top) if c is not candidates[i])
        trace = [
            f"реранкер пересортировал {moved} из {min(k, len(top))} позиций, "
            f"лучший скор={best:.3f}"
        ]

        abstain = None
        if self.threshold is not None:
            abstain = best < self.threshold
            trace.append(
                f"порог={self.threshold}: "
                + ("ответа в корпусе нет" if abstain else "ответ, похоже, есть")
            )

        return RetrievalResult(
            chunks=top,
            best_score=best,
            abstain_hint=abstain,
            trace=trace,
        )
