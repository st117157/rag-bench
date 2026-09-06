"""Шаг 2. Проверить набор вопросов.

Запуск:  python scripts/02_check_questions.py

Запускайте каждый раз, когда добавили или поправили вопросы.
Скрипт ловит четыре класса ошибок, которые иначе всплывут только
на финальном прогоне, когда переделывать уже дорого:

  1. gold_sources указывает на файл, которого нет в корпусе
     (опечатка в имени — самая частая ошибка);
  2. unanswerable-вопрос, у которого зачем-то есть источники;
  3. дубликаты id и пропущенные поля;
  4. САМОЕ ВАЖНОЕ: semantic-вопрос, который на самом деле находится
     обычным словарным поиском. Такой вопрос — замаскированный lexical,
     и он тихо ломает главный вывод бенчмарка.

Про четвёртую проверку стоит сказать отдельно, потому что она сделана
не так, как напрашивается.

Наивный вариант — посчитать, сколько слов вопроса встречается в документе.
Он плохо работает: документация многословна, и на любую тему найдётся
десяток общих слов. Полного нуля совпадений в живом тексте не добиться.

Поэтому мы проверяем напрямую то, что нас на самом деле волнует:
ЗАПУСКАЕМ BM25 по корпусу и смотрим, находит ли он нужный документ.
Если словарный поиск и так уверенно ставит правильный файл в топ —
вопрос не семантический, каким бы он ни выглядел на глаз.
Никаких эвристик по словам, прямое измерение.

Совпадения редких слов печатаются рядом как подсказка: обычно именно
одно-два таких слова и вытаскивают документ наверх, и переформулировать
достаточно только их.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from config import CORPUS_DIR, QUESTIONS_FILE  # noqa: E402

VALID_TYPES = {"lexical", "semantic", "multihop", "unanswerable"}

# Служебные слова не несут смысла и не считаются совпадением.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "without", "from", "by", "as", "is", "are", "was", "were",
    "be", "been", "do", "does", "did", "how", "what", "which", "when", "where",
    "why", "who", "can", "could", "should", "would", "will", "my", "me", "i",
    "it", "its", "that", "this", "these", "those", "there", "then", "than",
    "not", "no", "one", "all", "any", "some", "into", "out", "up", "down",
    "very", "only", "own", "same", "so", "too", "just", "now", "also", "get",
    "use", "using", "used", "make", "want", "need", "keep", "run", "runs",
    "have", "has", "had", "you", "your", "they", "them", "their", "we", "our",
    "app", "code", "file", "files", "way", "thing", "work", "works", "still",
}

# Если BM25 ставит нужный документ в топ-N, вопрос не семантический.
# 3 — разумный компромисс: попадание на 1-2 место означает уверенное
# словарное совпадение, а к пятому месту документ мог попасть случайно.
BM25_TOP_N = 3

# Слово, которое встречается почти во всех документах корпуса, не помогает
# поиску: по нему находится всё, а значит — ничего. Ровно это и выражает
# IDF внутри BM25. В подсказках показываем только СЛОВА-РЕДКОСТИ:
# те, что встречаются менее чем в COMMON_WORD_RATIO документов корпуса.
COMMON_WORD_RATIO = 0.25


def words(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
        if w not in STOPWORDS
    ]


def main() -> None:
    questions = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    corpus = {p.name: p.read_text(encoding="utf-8", errors="ignore").lower()
              for p in CORPUS_DIR.glob("*.md")}

    if not corpus:
        sys.exit("Корпус пуст. Сначала: python scripts/01_download_corpus.py")

    # Считаем, в скольких документах встречается каждое слово.
    # Это и есть document frequency — основа IDF в BM25.
    doc_freq: Counter[str] = Counter()
    for text in corpus.values():
        doc_freq.update(set(words(text)))
    common_cutoff = len(corpus) * COMMON_WORD_RATIO

    # BM25 по документам целиком (не по чанкам): нас интересует только,
    # находит ли словарный поиск нужный ФАЙЛ.
    from rank_bm25 import BM25Okapi

    names = list(corpus)
    bm25 = BM25Okapi([words(corpus[n]) for n in names])

    def bm25_rank(question: str, target: str) -> int:
        """Какое место занимает target в выдаче BM25. 1 — первое."""
        scores = bm25.get_scores(words(question))
        order = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
        return [names[i] for i in order].index(target) + 1

    errors: list[str] = []
    warnings: list[str] = []

    ids = Counter(q["id"] for q in questions)
    for dup, n in ids.items():
        if n > 1:
            errors.append(f"дубликат id: {dup} встречается {n} раз")

    for q in questions:
        qid = q.get("id", "<без id>")

        for field in ("id", "type", "question", "gold_answer"):
            if not q.get(field):
                errors.append(f"{qid}: не заполнено поле {field}")

        qtype = q.get("type")
        if qtype not in VALID_TYPES:
            errors.append(f"{qid}: неизвестный тип '{qtype}'")

        sources = q.get("gold_sources") or []

        for src in sources:
            if src not in corpus:
                errors.append(f"{qid}: файла '{src}' нет в корпусе")

        if qtype == "unanswerable" and sources:
            errors.append(f"{qid}: у unanswerable-вопроса не должно быть источников")

        if qtype != "unanswerable" and not sources:
            errors.append(f"{qid}: не указаны gold_sources")

        if qtype == "multihop" and len(sources) < 2:
            warnings.append(f"{qid}: multihop обычно требует минимум двух источников")

        # Главная проверка: не является ли semantic-вопрос переодетым lexical.
        if qtype == "semantic":
            qwords = set(words(q["question"]))
            for src in sources:
                if src not in corpus:
                    continue
                rank = bm25_rank(q["question"], src)
                if rank <= BM25_TOP_N:
                    rare = sorted(
                        (w for w in qwords
                         if w in corpus[src] and doc_freq[w] < common_cutoff),
                        key=lambda w: doc_freq[w],
                    )
                    detail = ", ".join(f"{w} ({doc_freq[w]} док.)" for w in rare[:4])
                    warnings.append(
                        f"{qid}: BM25 находит {src} на {rank}-м месте — "
                        f"это не семантический вопрос. Виноваты редкие слова: "
                        f"[{detail or 'нет явных'}]"
                    )

    types = Counter(q.get("type") for q in questions)
    print(f"Вопросов: {len(questions)}")
    for t in ("lexical", "semantic", "multihop", "unanswerable"):
        print(f"  {t:<13} {types.get(t, 0)}")

    if warnings:
        print(f"\nПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("\nОшибок нет.")
    if len(questions) < 50:
        print(f"Вопросов пока {len(questions)}, цель — 50.")


if __name__ == "__main__":
    main()
