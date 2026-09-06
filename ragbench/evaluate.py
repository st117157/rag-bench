"""Метрики. Считаем поиск и генерацию РАЗДЕЛЬНО.

Это главное методическое правило проекта. По одному итоговому
ответу невозможно понять, что именно сломалось: ретривер не нашёл
нужный документ или модель не смогла им воспользоваться.

Метрики поиска (дёшево, без вызовов LLM):
  recall@k  — доля вопросов, где хотя бы один нужный файл попал в топ-k
  mrr       — 1 / позиция первого правильного файла, усреднённая

Метрики ответа (дорого, через LLM-судью):
  correctness — совпадает ли ответ с эталоном по смыслу
  abstention  — правильно ли система молчит на вопросы без ответа
"""

from __future__ import annotations

from dataclasses import dataclass

from ragbench.llm import complete

JUDGE_SYSTEM = """You are a strict grader. You get a question,
a reference answer and a system answer.

Decide whether the system answer matches the reference IN MEANING.
Different wording is fine.
An answer that carries the correct substance plus extra detail is correct.
An answer that contradicts the reference or invents facts is incorrect.

Reply with one word: CORRECT or INCORRECT."""


# --- метрики поиска ---


def recall_at_k(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    """1.0, если хотя бы один эталонный файл найден."""
    if not gold_sources:          # вопрос типа unanswerable
        return float("nan")       # recall тут не определён, считаем отдельно
    return 1.0 if set(retrieved_sources) & set(gold_sources) else 0.0


def reciprocal_rank(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    """1 / позиция первого правильного источника (позиции с 1)."""
    if not gold_sources:
        return float("nan")
    for i, src in enumerate(retrieved_sources, start=1):
        if src in gold_sources:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    """Доля выданных источников, которые действительно нужны.

    Recall отвечает на вопрос «нашли ли нужное», precision — «сколько
    лишнего приехало вместе с нужным». Пара этих цифр рядом объясняет
    почти всё в таблице результатов.

    Мусор в контексте не бесплатен дважды: за него платят токенами
    и им же сбивают модель с толку. Именно поэтому long-context
    показывает отличный recall при precision, близком к нулю:
    он «находит» ответ ровно в том смысле, в каком его находит
    человек, которому выдали всю библиотеку целиком.
    """
    if not gold_sources:
        return float("nan")     # для unanswerable понятие не определено
    if not retrieved_sources:
        return 0.0
    hits = sum(1 for s in retrieved_sources if s in gold_sources)
    return hits / len(retrieved_sources)


# --- метрики ответа ---


@dataclass
class Judgement:
    correct: bool
    input_tokens: int
    output_tokens: int


def judge_answer(question: str, gold_answer: str, system_answer: str) -> Judgement:
    """LLM-судья. Вызывается один раз на вопрос на подход.

    Судья тоже ошибается, поэтому часть его вердиктов проверена
    вручную: 20 карточек, разбор — в judge_check.md, совпадение 85%.
    Без такой проверки цифрам `correct` верить нельзя.
    """
    prompt = (
        f"QUESTION: {question}\n\n"
        f"REFERENCE ANSWER: {gold_answer}\n\n"
        f"SYSTEM ANSWER: {system_answer}"
    )
    resp = complete(prompt, system=JUDGE_SYSTEM, max_tokens=10)
    verdict = resp.text.strip().upper()
    return Judgement(
        correct=verdict.startswith("CORRECT"),
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


def abstention_correct(is_unanswerable: bool, abstained: bool) -> bool:
    """Правильное поведение: молчать там, где ответа нет, и только там."""
    return is_unanswerable == abstained
