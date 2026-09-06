"""Подходы 5 и 6: длинный контекст. Векторов нет.

Главное число, видное уже на первом запуске: корпус — около 1.1 МБ
текста, примерно 190 тысяч токенов. В окно gpt-4o-mini
(128k) он не помещается. Это не поломка проекта, а его результат:
long-context не масштабируется линейно, и упереться в стену на корпусе
среднего размера — нормальный исход, который надо честно показать.

Реализованы два варианта.

LongContextRetriever — «в лоб». Складываем в промпт столько корпуса,
сколько влезает в MAX_CONTEXT_TOKENS, остальное отбрасываем.
Нужен как честная демонстрация проблемы: recall на том, что влезло,
отличный, а на остальном — нулевой, и precision близок к нулю всегда.

LongContextFiltered — «грубый фильтр плюс целые документы». Сначала
дешёвым лексическим поиском отбираем несколько подходящих ФАЙЛОВ,
затем подаём их ЦЕЛИКОМ, не разрезая на чанки. Векторов по-прежнему нет.

Второй вариант интересен тем, что снимает главную слабость обычного RAG:
чанк — это обрывок, у которого отрезано начало мысли, а целый документ
самодостаточен. При этом фильтрация почти бесплатна.
"""

from __future__ import annotations

from config import LONGCTX_FILES, MAX_CONTEXT_TOKENS
from ragbench.corpus import Chunk
from ragbench.llm import count_tokens
from ragbench.retrievers.base import BaseRetriever, RetrievalResult


class LongContextRetriever(BaseRetriever):
    """Весь корпус в промпт, поиска нет вообще."""

    name = "longctx"

    def __init__(self, chunks) -> None:
        super().__init__(chunks)
        self._budgeted: list[Chunk] = []
        self._total_tokens = 0
        self._truncated = False

    def build(self) -> None:
        """Один раз посчитать, что вообще влезает в бюджет."""
        used = 0
        for chunk in self.chunks:
            n = count_tokens(chunk.text)
            if used + n > MAX_CONTEXT_TOKENS:
                self._truncated = True
                break
            self._budgeted.append(chunk)
            used += n
        self._total_tokens = used

        share = len(self._budgeted) / len(self.chunks) if self.chunks else 0
        print(
            f"[longctx] в бюджет {MAX_CONTEXT_TOKENS} токенов влезло "
            f"{len(self._budgeted)} чанков из {len(self.chunks)} "
            f"({share:.0%}, {used} токенов)"
        )
        if self._truncated:
            print(
                "[longctx] ВНИМАНИЕ: корпус обрезан. Часть вопросов этот подход "
                "не сможет ответить в принципе — отметьте это в README."
            )

    def search(self, question: str, k: int) -> RetrievalResult:
        # Поиска нет: на любой вопрос отдаём один и тот же контекст.
        # Параметр k игнорируется намеренно.
        return RetrievalResult(
            chunks=self._budgeted,
            trace=[f"без поиска, {self._total_tokens} токенов в промпте"],
        )


class LongContextFilteredRetriever(BaseRetriever):
    """Грубый отбор нескольких файлов, дальше документы целиком."""

    name = "longctx_filtered"

    def __init__(self, chunks, n_files: int | None = None) -> None:
        super().__init__(chunks)
        self.n_files = n_files or LONGCTX_FILES
        self.bm25 = None
        self.docs: dict[str, list[Chunk]] = {}
        self.names: list[str] = []

    def build(self) -> None:
        """Собрать чанки обратно в документы и построить лексический индекс.

        Индексируются ДОКУМЕНТЫ, а не чанки: задача этого этапа —
        не найти нужный абзац, а всего лишь отсеять заведомо лишние файлы.
        Такой индекс строится за доли секунды.
        """
        from rank_bm25 import BM25Okapi

        from ragbench.retrievers.hybrid import tokenize

        for chunk in self.chunks:
            self.docs.setdefault(chunk.source, []).append(chunk)

        self.names = sorted(self.docs)
        self.bm25 = BM25Okapi(
            [tokenize(" ".join(c.text for c in self.docs[n])) for n in self.names]
        )

    def search(self, question: str, k: int) -> RetrievalResult:
        from ragbench.retrievers.hybrid import tokenize

        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(self.names)), key=lambda i: scores[i], reverse=True)

        picked: list[Chunk] = []
        used_files: list[str] = []
        used_tokens = 0

        for idx in best[: self.n_files]:
            name = self.names[idx]
            doc_chunks = self.docs[name]
            doc_tokens = sum(count_tokens(c.text) for c in doc_chunks)
            if used_tokens + doc_tokens > MAX_CONTEXT_TOKENS:
                break
            picked += doc_chunks          # документ целиком, без нарезки
            used_files.append(name)
            used_tokens += doc_tokens

        return RetrievalResult(
            chunks=picked,
            trace=[
                f"отобрано файлов: {len(used_files)} из {len(self.names)}, "
                f"{used_tokens} токенов; {', '.join(used_files[:3])}"
            ],
        )
