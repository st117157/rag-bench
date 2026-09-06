"""Шаг 3. Подобрать порог отказа для реранкера.

Запуск:  python scripts/03_calibrate_threshold.py

Обращений к API не делает — денег не стоит.

ЗАЧЕМ ЭТО НУЖНО

Retrieval устроен так, что всегда что-то возвращает: ближайшие соседи
существуют и тогда, когда в корпусе нет ничего подходящего. Модель
получает пять правдоподобных фрагментов и сочиняет по ним ответ.
Это главный источник галлюцинаций в RAG, и лечится он не промптом,
а измерением.

Cross-encoder выдаёт абсолютную оценку релевантности пары
(вопрос, фрагмент). Значит, у неё есть смысл сама по себе, и её можно
сравнить с порогом: лучший кандидат ниже порога — ответа в корпусе нет.

Скрипт считает эти оценки для всех вопросов, перебирает пороги
и находит тот, при котором система чаще всего права в решении
«отвечать или молчать».

КАК ЧИТАТЬ РЕЗУЛЬТАТ

Печатается таблица: порог, сколько раз система правильно промолчала
на unanswerable, сколько раз ошибочно промолчала на отвечаемом вопросе,
и итоговая доля правильных решений.

Обратите внимание на компромисс. Слишком высокий порог — система молчит
даже там, где ответ есть (осторожная и бесполезная). Слишком низкий —
отвечает всегда, включая выдумки (уверенная и опасная). Оптимум зависит
от того, что для вас дороже, и это решение продуктовое, а не техническое.
Напишите об этом в README отдельным абзацем — такие рассуждения ценят
выше, чем сами цифры.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from config import QUESTIONS_FILE, TOP_K  # noqa: E402
from ragbench.corpus import build_chunks  # noqa: E402
from ragbench.retrievers.reranked import RerankedRetriever  # noqa: E402


def main() -> None:
    questions = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    chunks = build_chunks()
    print(f"Вопросов: {len(questions)}, чанков: {len(chunks)}")

    # threshold=None означает «порог не применять» — сейчас нам нужны
    # только сырые оценки, решение будем принимать сами.
    retriever = RerankedRetriever(chunks, threshold=None)
    print("Строю индекс...")
    retriever.build()

    print("Считаю оценки релевантности...")
    scored: list[tuple[str, bool, float]] = []
    for q in questions:
        r = retriever.search(q["question"], TOP_K)
        if r.best_score is None:
            continue
        scored.append((q["id"], q.get("type") == "unanswerable", r.best_score))

    answerable = [s for _, u, s in scored if not u]
    unanswerable = [s for _, u, s in scored if u]

    if not unanswerable:
        sys.exit("В наборе нет вопросов типа unanswerable — калибровать не по чему.")

    print(
        f"\nОтвечаемые вопросы ({len(answerable)}): "
        f"скор от {min(answerable):.2f} до {max(answerable):.2f}, "
        f"медиана {sorted(answerable)[len(answerable) // 2]:.2f}"
    )
    print(
        f"Без ответа в корпусе ({len(unanswerable)}): "
        f"скор от {min(unanswerable):.2f} до {max(unanswerable):.2f}, "
        f"медиана {sorted(unanswerable)[len(unanswerable) // 2]:.2f}"
    )

    # Кандидаты в пороги — середины между соседними наблюдаемыми значениями.
    values = sorted({round(s, 2) for _, _, s in scored})
    candidates = [
        round((a + b) / 2, 3) for a, b in zip(values, values[1:], strict=False)
    ]

    print(f"\n{'порог':>8} {'молчит верно':>13} {'молчит зря':>11} {'точность':>9}")
    print("-" * 45)

    best_threshold, best_acc = None, -1.0
    rows = []
    for t in candidates:
        # Правильное поведение: молчать на unanswerable и только на нём.
        correct_silence = sum(1 for _, u, s in scored if u and s < t)
        wrong_silence = sum(1 for _, u, s in scored if not u and s < t)
        accuracy = (
            correct_silence + (len(answerable) - wrong_silence)
        ) / len(scored)
        rows.append((t, correct_silence, wrong_silence, accuracy))
        if accuracy > best_acc:
            best_threshold, best_acc = t, accuracy

    # Печатаем не всё подряд, а окрестность оптимума — иначе таблица
    # на сто строк, в которой ничего не видно.
    rows.sort(key=lambda r: r[3], reverse=True)
    for t, cs, ws, acc in rows[:12]:
        print(f"{t:>8.2f} {cs:>7}/{len(unanswerable):<5} {ws:>7}/{len(answerable):<3} {acc:>8.0%}")

    print(f"\nЛучший порог по точности: {best_threshold}  ({best_acc:.0%})")

    baseline = len(answerable) / len(scored)
    print(f"Стратегия «никогда не молчать» даёт при этом: {baseline:.0%}")
    if best_acc <= baseline + 0.01:
        print(
            "\nВНИМАНИЕ: порог не обыгрывает бездействие. Но это, скорее всего,\n"
            "недостаток метрики, а не механизма: классы несбалансированы\n"
            "(отвечаемых вопросов намного больше), и accuracy вознаграждает\n"
            "большинство. К тому же отказ на вопросе, где ретривер и так\n"
            "ничего не нашёл, здесь считается ошибкой, хотя ошибкой не является.\n"
            "\nЗапустите разбор по трём категориям, он показывает картину честнее:\n"
            "    python scripts/05_analyze_abstention.py --method reranked"
        )
    else:
        print("\nЧтобы включить порог, впишите в config.py:")
        print(f'    ABSTAIN_THRESHOLDS["reranked"] = {best_threshold}')


if __name__ == "__main__":
    main()
