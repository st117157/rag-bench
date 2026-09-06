"""Подход 1: наивный RAG. Только плотные векторы (эмбеддинги).

Это тот самый «RAG из туториала 2023 года»: порезали на чанки,
посчитали эмбеддинги, ищем ближайших соседей, отдаём топ-5.
Он здесь как базовая линия — точка отсчёта, относительно которой
всё остальное должно быть лучше. Если что-то оказалось хуже наивного
подхода, это тоже результат, и его надо честно показать.
"""

from __future__ import annotations

import numpy as np

from config import EMBEDDING_MODEL, INDEX_DIR
from ragbench.retrievers.base import BaseRetriever, RetrievalResult


class NaiveDenseRetriever(BaseRetriever):
    name = "naive"

    def __init__(self, chunks) -> None:
        super().__init__(chunks)
        self.model = None
        self.index = None

    def build(self) -> None:
        """Посчитать эмбеддинги всех чанков и сложить в FAISS.

        Считается один раз и кэшируется на диск: на 155 файлах это
        несколько минут на CPU, и второй раз ждать не хочется.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cache = INDEX_DIR / "dense.npy"

        if cache.exists():
            embeddings = np.load(cache)
            # Простая защита от рассинхрона: если чанков стало больше/меньше,
            # кэш невалиден. Без этой проверки рассинхрон проявится
            # как необъяснимо плохие метрики.
            if len(embeddings) != len(self.chunks):
                cache.unlink()
                embeddings = None
        else:
            embeddings = None

        if embeddings is None:
            texts = [c.text for c in self.chunks]
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,   # важно: с нормализацией
            ).astype("float32")              # скалярное произведение = косинус
            np.save(cache, embeddings)

        import faiss

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question: str, k: int) -> RetrievalResult:
        if self.index is None:
            raise RuntimeError("Сначала вызовите build()")

        q_vec = self.model.encode(
            [question], normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(q_vec, k)

        found = [self.chunks[i] for i in ids[0] if i != -1]
        return RetrievalResult(
            chunks=found,
            trace=[f"dense top-{k}, best score={scores[0][0]:.3f}"],
        )


# Возможное улучшение: заменить локальную модель на эмбеддинги через API
# (например, text-embedding-3-small). Качество должно вырасти заметно,
# но появится цена за индексацию, и её нужно будет включить в таблицу
# стоимости — иначе сравнение с бесплатными подходами станет нечестным.
