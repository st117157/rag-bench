"""Загрузка корпуса и нарезка на чанки.

Самый недооценённый файл проекта. Качество RAG процентов на семьдесят
определяется тем, как порезаны документы, а не выбором векторной базы.

Здесь реализованы ДВЕ стратегии, и переключаются они одной строкой
в config.py. Это сделано специально: сравнить их между собой —
отдельный эксперимент, и его результат обычно впечатляет сильнее,
чем разница между самими подходами к поиску.

  "chars"    наивная нарезка по символам с нахлёстом.
             Быстро, тупо, рвёт код и таблицы посередине.

  "markdown" нарезка по заголовкам с сохранением цепочки заголовков
             в тексте чанка. Чанк перестаёт быть обрывком и становится
             самостоятельным фрагментом, у которого понятно происхождение.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_STRATEGY, CORPUS_DIR

# Заголовок markdown: "## Query Parameters { #query-parameters }"
# Якорь в фигурных скобках нужно отбросить, он мусорный для поиска.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{[^}]*\})?\s*$")


@dataclass
class Chunk:
    """Один кусок документа — минимальная единица поиска."""

    id: str            # уникальный идентификатор, "tutorial__body.md#3"
    text: str          # сам текст куска
    source: str        # имя файла-источника
    position: int      # порядковый номер куска внутри файла
    heading_path: str = ""   # "Request Body > Create your data model"
    meta: dict = field(default_factory=dict)

    def with_header(self) -> str:
        """Текст с приклеенным происхождением.

        Мелкий приём с заметным эффектом: модель видит, откуда взят
        фрагмент, и реже путает похожие разделы между собой.
        """
        head = f"[source: {self.source}]"
        if self.heading_path:
            head += f"\n[section: {self.heading_path}]"
        return f"{head}\n{self.text}"


def load_documents(corpus_dir: Path = CORPUS_DIR) -> dict[str, str]:
    """Прочитать все .md файлы корпуса. Возвращает {имя файла: текст}."""
    docs: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs[path.name] = text
    if not docs:
        raise FileNotFoundError(
            f"В {corpus_dir} нет .md файлов. "
            "Сначала запустите: python scripts/01_download_corpus.py"
        )
    return docs


# ---------------------------------------------------------------------
# Стратегия 1: нарезка по символам
# ---------------------------------------------------------------------


def chunk_by_chars(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, str]]:
    """Наивная нарезка. Возвращает список (текст, путь заголовков).

    Нахлёст нужен, чтобы предложение, разрезанное пополам, целиком
    попало хотя бы в один чанк. Путь заголовков здесь всегда пустой —
    эта стратегия про структуру документа ничего не знает.
    """
    if overlap >= size:
        raise ValueError("overlap должен быть меньше size")
    if len(text) <= size:
        return [(text, "")]

    pieces: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        pieces.append((text[start : start + size], ""))
        start += size - overlap
    return pieces


# ---------------------------------------------------------------------
# Стратегия 2: нарезка по заголовкам markdown
# ---------------------------------------------------------------------


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Разбить документ на разделы по заголовкам.

    Идём по строкам, ведём стек текущих заголовков. Встретили "##" —
    закрыли предыдущий раздел, обновили стек на нужном уровне.
    В итоге каждый раздел знает свой полный путь:
    "Request Body > Create your data model".
    """
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((body, " > ".join(stack)))
        buf.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Обрезаем стек до текущего уровня и кладём новый заголовок.
            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")     # пропущенный уровень, бывает в живых текстах
            stack.append(title)
        else:
            buf.append(line)

    flush()
    return sections


def chunk_by_markdown(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, str]]:
    """Нарезка по заголовкам с досечкой слишком длинных разделов.

    Три шага:
      1. режем документ на разделы по заголовкам;
      2. раздел короче size — оставляем целиком, это идеальный чанк;
      3. раздел длиннее — досекаем по абзацам (пустая строка),
         а если и абзац огромный, отдаём его нарезке по символам.

    Ключевое отличие от стратегии "chars": граница чанка почти всегда
    совпадает со смысловой границей текста, и у каждого чанка есть
    путь заголовков, который потом приклеивается к тексту в промпте.
    """
    out: list[tuple[str, str]] = []

    for body, path in split_by_headings(text):
        if len(body) <= size:
            out.append((body, path))
            continue

        # Раздел слишком длинный — собираем из абзацев куски до size.
        current: list[str] = []
        current_len = 0
        for para in body.split("\n\n"):
            if len(para) > size:
                if current:
                    out.append(("\n\n".join(current), path))
                    current, current_len = [], 0
                out += [(piece, path) for piece, _ in chunk_by_chars(para, size, overlap)]
                continue

            if current_len + len(para) > size and current:
                out.append(("\n\n".join(current), path))
                current, current_len = [], 0

            current.append(para)
            current_len += len(para) + 2

        if current:
            out.append(("\n\n".join(current), path))

    return out or [(text, "")]


# ---------------------------------------------------------------------


STRATEGIES = {
    "chars": chunk_by_chars,
    "markdown": chunk_by_markdown,
}


def build_chunks(
    docs: dict[str, str] | None = None, strategy: str | None = None
) -> list[Chunk]:
    """Превратить документы в плоский список чанков."""
    if docs is None:
        docs = load_documents()

    strategy = strategy or CHUNK_STRATEGY
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Неизвестная стратегия '{strategy}'. Доступны: {list(STRATEGIES)}"
        )
    splitter = STRATEGIES[strategy]

    chunks: list[Chunk] = []
    for source, text in docs.items():
        for i, (piece, path) in enumerate(splitter(text)):
            chunks.append(
                Chunk(
                    id=f"{source}#{i}",
                    text=piece,
                    source=source,
                    position=i,
                    heading_path=path,
                )
            )
    return chunks


if __name__ == "__main__":
    # Сравнение двух стратегий: python -m ragbench.corpus
    docs = load_documents()
    print(f"Документов: {len(docs)}\n")

    for name in STRATEGIES:
        cs = build_chunks(docs, strategy=name)
        lengths = [len(c.text) for c in cs]
        with_path = sum(1 for c in cs if c.heading_path)
        print(f"стратегия {name!r}")
        print(f"  чанков:          {len(cs)}")
        print(f"  средняя длина:   {sum(lengths) / len(lengths):.0f} символов")
        print(f"  самый короткий:  {min(lengths)}")
        print(f"  самый длинный:   {max(lengths)}")
        print(f"  знают свой раздел: {with_path} ({with_path / len(cs):.0%})")
        print()

    print("Пример чанка при стратегии 'markdown':\n")
    example = [c for c in build_chunks(docs, "markdown") if c.heading_path][3]
    print(example.with_header()[:500])
