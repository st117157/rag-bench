"""Подход 7: гибрид для поиска, cross-encoder только как привратник.

Идея выросла прямо из результатов бенчмарка, а не из литературы.

Наблюдение было такое. Реранкер как ранжировщик проиграл: recall 0.725
против 0.85 у гибрида, и при этом латентность в 27 раз выше. Но как
детектор «ответа в корпусе нет» он оказался полезен — при пороге 0.0
ловил 17 случаев, где отвечать было нечем, ценой 4 потерянных ответов.

Отсюда очевидный вопрос: зачем позволять ему переставлять выдачу, если
в этом он плох? Возьмём у него только то, в чём он хорош.

Как устроено:
  1. порядок документов определяет гибрид, и он остаётся нетронутым;
  2. cross-encoder оценивает ТОЛЬКО отобранные k чанков — не 20
     кандидатов, как в reranked, а пять;
  3. его оценка используется ровно для одного решения: отвечать или
     промолчать.

Что должно получиться: recall и MRR как у гибрида, механизм честного
отказа как у реранкера, а латентность где-то посередине — cross-encoder
считает вчетверо меньше пар, чем в полном реранкинге.

ВАЖНО ПРО ПОРОГ. Значение, подобранное для reranked, здесь не подходит:
там максимум берётся по CANDIDATES_K кандидатам, здесь по TOP_K, и
максимум по двадцати систематически выше максимума по пяти. Поэтому
пороги хранятся раздельно в ABSTAIN_THRESHOLDS. Откалибруйте свой:

    python scripts/05_analyze_abstention.py --method hybrid_gated
"""

from __future__ import annotations

from config import ABSTAIN_THRESHOLDS, RERANKER_MODEL
from ragbench.retrievers.base import FROM_CONFIG, RetrievalResult
from ragbench.retrievers.hybrid import HybridRetriever


class HybridGatedRetriever(HybridRetriever):
    name = "hybrid_gated"

    def __init__(self, chunks, alpha: float | None = None,
                 threshold=FROM_CONFIG) -> None:
        super().__init__(chunks, alpha)
        self.gate = None
        self.threshold = (
            ABSTAIN_THRESHOLDS.get(self.name)
            if threshold is FROM_CONFIG
            else threshold
        )

    def build(self) -> None:
        from sentence_transformers import CrossEncoder

        super().build()
        self.gate = CrossEncoder(RERANKER_MODEL)

    def search(self, question: str, k: int) -> RetrievalResult:
        # Поиск и порядок — целиком за гибридом.
        result = super().search(question, k)
        if not result.chunks:
            return RetrievalResult(chunks=[], abstain_hint=True)

        # Cross-encoder смотрит только на то, что уже отобрано.
        # Порядок при этом НЕ меняется — нас интересует одно число.
        scores = self.gate.predict([(question, c.text) for c in result.chunks])
        best = float(max(scores))

        trace = list(result.trace)
        trace.append(f"привратник оценил {len(result.chunks)} чанков, лучший={best:.3f}")

        abstain = None
        if self.threshold is not None:
            abstain = best < self.threshold
            trace.append(
                f"порог={self.threshold}: "
                + ("ответа в корпусе нет" if abstain else "ответ, похоже, есть")
            )

        return RetrievalResult(
            chunks=result.chunks,        # порядок гибрида сохранён
            best_score=best,
            abstain_hint=abstain,
            trace=trace,
        )
