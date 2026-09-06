"""Подход 4: агентный поиск грепом. Векторной БД нет вообще.

Это и есть «проект B». Идея, которую сейчас активно продвигают
(и по которой, например, работает Claude Code): вместо того чтобы
заранее считать эмбеддинги, дать модели инструменты поиска
и позволить ей самой решать, что искать и что открыть.

Цикл работы:
  1. модель смотрит на вопрос и оглавление корпуса;
  2. придумывает поисковые запросы (регулярки / ключевые слова);
  3. мы прогоняем их по файлам и возвращаем совпадения с контекстом;
  4. модель решает: хватит данных или искать дальше;
  5. повторяем до MAX_STEPS.

Главное преимущество перед векторами — отлаживаемость. Провал grep
виден целиком: искали "background task", нашли ноль. Провал
эмбеддинга не виден никак.
Главная слабость — синонимы. Для этого в наборе вопросов есть тип
`semantic`: там подход должен проигрывать, и это расхождение —
одно из ключевых наблюдений бенчмарка.
"""

from __future__ import annotations

import json
import re

from config import CORPUS_DIR
from ragbench.corpus import Chunk
from ragbench.llm import complete
from ragbench.retrievers.base import BaseRetriever, RetrievalResult

MAX_STEPS = 3            # сколько раундов поиска разрешаем агенту
MAX_MATCHES_PER_QUERY = 8
CONTEXT_CHARS = 600      # сколько символов вокруг совпадения отдаём

PLANNER_SYSTEM = """Ты — поисковый агент по документации FastAPI.
У тебя есть только полнотекстовый поиск по файлам (как grep).
Твоя задача — придумать поисковые запросы, которые найдут ответ.

Правила:
- запросы должны быть теми словами, которые РЕАЛЬНО написаны в документации,
  а не словами из вопроса пользователя;
- давай 2-4 запроса за раз, от точного к общему;
- отвечай ТОЛЬКО валидным JSON вида {"queries": ["...", "..."], "done": false}
- поставь "done": true, если считаешь, что найденного уже достаточно."""


class GrepAgentRetriever(BaseRetriever):
    name = "grep"

    def __init__(self, chunks) -> None:
        super().__init__(chunks)
        self._ce = None

    def build(self) -> None:
        # Индекса нет — в этом весь смысл подхода: время подготовки
        # здесь 0 секунд против минут у векторных подходов.
        #
        # Cross-encoder намеренно НЕ грузится здесь: он нужен только
        # для сортировки уже найденного, никакой предварительной
        # обработки корпуса не делает, и загружать его в build() значило бы
        # приписать этому подходу чужое время индексации.
        return None

    def _reranker(self):
        """Ленивая загрузка cross-encoder при первом обращении."""
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            from config import RERANKER_MODEL

            self._ce = CrossEncoder(RERANKER_MODEL)
        return self._ce

    # --- инструмент, который мы даём модели ---

    def _grep(self, pattern: str) -> list[tuple[str, int, str]]:
        """Найти pattern во всех файлах корпуса.

        Возвращает список (имя файла, позиция, фрагмент вокруг совпадения),
        отсортированный так, что файлы с бОльшим числом совпадений идут выше.

        Тонкий момент, на котором легко ошибиться: нельзя просто обходить
        файлы по алфавиту и останавливаться на первых N совпадениях —
        тогда результат зависит от имён файлов, а не от релевантности,
        и подход проиграет незаслуженно. Поэтому сначала считаем совпадения
        во ВСЕХ файлах, и только потом отбираем лучшие. Плюс берём не больше
        PER_FILE_LIMIT фрагментов из одного файла, чтобы один многословный
        документ не занял всю выдачу.
        """
        PER_FILE_LIMIT = 2

        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Модель вполне может выдать сломанную регулярку —
            # обрабатываем как обычный текст.
            rx = re.compile(re.escape(pattern), re.IGNORECASE)

        # {имя файла: (сколько всего совпадений, [фрагменты])}
        per_file: dict[str, tuple[int, list[tuple[int, str]]]] = {}

        for path in sorted(CORPUS_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            snippets: list[tuple[int, str]] = []
            total = 0
            for m in rx.finditer(text):
                total += 1
                if len(snippets) < PER_FILE_LIMIT:
                    start = max(0, m.start() - CONTEXT_CHARS // 2)
                    snippets.append((start, text[start : start + CONTEXT_CHARS]))
            if total:
                per_file[path.name] = (total, snippets)

        ranked = sorted(per_file.items(), key=lambda kv: kv[1][0], reverse=True)

        hits: list[tuple[str, int, str]] = []
        for fname, (_, snippets) in ranked:
            for pos, snippet in snippets:
                hits.append((fname, pos, snippet))
                if len(hits) >= MAX_MATCHES_PER_QUERY:
                    return hits
        return hits

    def _file_listing(self) -> str:
        """Оглавление корпуса — дешёвая и очень полезная подсказка агенту.

        Имена файлов вида tutorial__security__oauth2-jwt.md сами по себе
        несут смысл. Часто агент находит нужный файл прямо по имени,
        не делая ни одного поиска.
        """
        names = sorted(p.name for p in CORPUS_DIR.glob("*.md"))
        return "\n".join(names)

    # --- основной цикл ---

    def search(self, question: str, k: int) -> RetrievalResult:
        collected: dict[str, Chunk] = {}
        trace: list[str] = []
        in_tok = out_tok = 0
        history = ""

        for step in range(MAX_STEPS):
            prompt = (
                f"Вопрос пользователя: {question}\n\n"
                f"Файлы в корпусе:\n{self._file_listing()}\n\n"
                f"История поиска:\n{history or '(пусто, это первый шаг)'}\n\n"
                "Какие запросы искать дальше?"
            )
            resp = complete(prompt, system=PLANNER_SYSTEM, max_tokens=300)
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens

            queries, done = _parse_plan(resp.text)
            trace.append(f"шаг {step + 1}: запросы {queries}")

            step_log: list[str] = []
            for q in queries:
                hits = self._grep(q)
                step_log.append(f"'{q}' -> {len(hits)} совпадений")
                for fname, pos, snippet in hits:
                    cid = f"{fname}#g{pos}"
                    if cid not in collected:
                        collected[cid] = Chunk(
                            id=cid, text=snippet, source=fname, position=pos
                        )

            history += "\n".join(step_log) + "\n"
            if done or len(collected) >= k * 3:
                break

        # Собранные куски идут в том порядке, в каком нашлись, а это
        # не порядок релевантности. Остальные подходы отдают
        # отсортированный список, поэтому сравнение было бы нечестным.
        # Пересортировываем тем же cross-encoder — и получаем связку
        # grep + rerank, в которой по-прежнему нет ни одного эмбеддинга.
        found = list(collected.values())
        best = None
        if found:
            scores = self._reranker().predict([(question, c.text) for c in found])
            ordered = sorted(
                zip(found, scores, strict=True), key=lambda p: p[1], reverse=True
            )
            found = [c for c, _ in ordered[:k]]
            best = float(ordered[0][1])
            trace.append(f"реранкинг {len(collected)} найденных, лучший скор={best:.3f}")

        return RetrievalResult(
            chunks=found,
            best_score=best,
            extra_input_tokens=in_tok,
            extra_output_tokens=out_tok,
            trace=trace,
        )


def _parse_plan(raw: str) -> tuple[list[str], bool]:
    """Достать JSON из ответа модели, не падая на мусоре вокруг."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return [], True
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return [], True
    queries = [q for q in data.get("queries", []) if isinstance(q, str)][:4]
    return queries, bool(data.get("done", False))
