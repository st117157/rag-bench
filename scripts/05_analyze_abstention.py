"""Шаг 5. Разобраться, чего на самом деле стоит порог отказа.

Запуск:  python scripts/05_analyze_abstention.py
         python scripts/05_analyze_abstention.py --method hybrid_gated
         python scripts/05_analyze_abstention.py --method reranked -3.5 -2.0 0.0

Порог у разных подходов разный, потому что оценка берётся по разному
числу кандидатов: reranked считает максимум по 20, hybrid_gated — по 5.
Каждый подход калибруется отдельно, чужое значение не подойдёт.

Обращений к API не делает.

ЗАЧЕМ

Скрипт 03 считает точность решения «отвечать или молчать» и упирается
в проблему несбалансированных классов: отвечаемых вопросов вчетверо
больше, поэтому стратегия «никогда не молчать» уже даёт 80%, и любой
порог выглядит бесполезным.

Но там спрятано более тонкое обстоятельство. Отказ считается ошибочным,
если вопрос отвечаемый. А что, если по этому вопросу ретривер всё равно
ничего не нашёл? Тогда системе нечем отвечать, и промолчать — правильно.
Такой отказ ошибкой не является, хотя скрипт 03 засчитывает его как ошибку.

Этот скрипт делит отказы на три группы:

  ОПРАВДАННЫЙ   вопрос без ответа в корпусе — промолчать верно
  ПОЛЕЗНЫЙ      вопрос отвечаемый, но ретривер промахнулся (recall = 0),
                отвечать всё равно было бы нечем — молчание спасает
                от выдумки
  ВРЕДНЫЙ        нужный документ найден, а система молчит — вот это
                настоящая потеря

Разделение меняет картину: то, что выглядело как «механизм не окупается»,
может оказаться «механизм ловит галлюцинации почти бесплатно».

ЧТО ДЕЛАТЬ С РЕЗУЛЬТАТОМ

Рабочий порог — тот, у которого мало ВРЕДНЫХ отказов при большом числе
ОПРАВДАННЫХ и ПОЛЕЗНЫХ.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from config import QUESTIONS_FILE, RESULTS_DIR, TOP_K  # noqa: E402
from ragbench.corpus import build_chunks  # noqa: E402
from ragbench.evaluate import recall_at_k  # noqa: E402
from ragbench.retrievers import REGISTRY  # noqa: E402

DEFAULT_THRESHOLDS = [-9.0, -7.0, -5.0, -4.0, -3.5, -3.0, -2.0, -1.0, 0.0, 1.0]

# Подходы, у которых есть абсолютная оценка релевантности (best_score).
# У остальных её нет: косинус и BM25 дают относительные величины,
# сравнивать их с порогом бессмысленно.
SCORED_METHODS = ("reranked", "hybrid_gated", "grep")


def main() -> None:
    args = sys.argv[1:]

    method = "reranked"
    if "--method" in args:
        i = args.index("--method")
        method = args[i + 1]
        del args[i : i + 2]

    if method not in SCORED_METHODS:
        sys.exit(
            f"У подхода '{method}' нет абсолютной оценки релевантности.\n"
            f"Калибровать можно только: {', '.join(SCORED_METHODS)}"
        )

    thresholds = [float(t) for t in args] or DEFAULT_THRESHOLDS

    questions = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    chunks = build_chunks()
    print(f"Подход: {method}")
    print(f"Вопросов: {len(questions)}, чанков: {len(chunks)}")

    # threshold=None: сейчас нужны сырые оценки, решение принимаем сами.
    retriever = REGISTRY[method](chunks, threshold=None)
    print("Строю индекс...")
    retriever.build()

    print("Считаю оценки и recall по каждому вопросу...\n")
    records = []
    for q in questions:
        r = retriever.search(q["question"], TOP_K)
        gold = q.get("gold_sources") or []
        unanswerable = q.get("type") == "unanswerable"
        found = (not unanswerable) and recall_at_k(r.sources, gold) == 1.0
        records.append(
            {
                "id": q["id"],
                "unanswerable": unanswerable,
                "found": found,          # нужный документ реально в выдаче
                "score": r.best_score if r.best_score is not None else -99.0,
            }
        )

    n_unans = sum(1 for x in records if x["unanswerable"])
    n_found = sum(1 for x in records if x["found"])
    n_missed = len(records) - n_unans - n_found

    print(f"Всего вопросов:                       {len(records)}")
    print(f"  без ответа в корпусе:               {n_unans}")
    print(f"  отвечаемые, документ найден:        {n_found}")
    print(f"  отвечаемые, ретривер промахнулся:   {n_missed}")
    print(
        "\nПоследняя группа — ключевая. На этих вопросах системе нечем "
        "отвечать,\nи молчание для них лучше выдумки."
    )

    print(f"\n{'порог':>7} {'оправдан':>9} {'полезен':>9} {'ВРЕДЕН':>8} "
          f"{'выдумок пропущено':>19}")
    print("-" * 60)

    rows = []
    for t in thresholds:
        justified = sum(1 for x in records if x["unanswerable"] and x["score"] < t)
        useful = sum(
            1 for x in records
            if not x["unanswerable"] and not x["found"] and x["score"] < t
        )
        harmful = sum(1 for x in records if x["found"] and x["score"] < t)
        missed_fabrications = n_unans - justified

        rows.append((t, justified, useful, harmful, missed_fabrications))
        print(
            f"{t:>7.2f} {justified:>5}/{n_unans:<3} {useful:>5}/{n_missed:<3} "
            f"{harmful:>4}/{n_found:<3} {missed_fabrications:>15}"
        )

    print(
        "\nоправдан — промолчал там, где ответа в корпусе нет"
        "\nполезен  — промолчал там, где сам ничего не нашёл"
        "\nВРЕДЕН   — промолчал, хотя нужный документ был в выдаче"
    )

    # Порог, при котором ещё нет вредных отказов, но уже есть польза.
    safe = [r for r in rows if r[3] == 0]
    if safe:
        best = max(safe, key=lambda r: r[1] + r[2])
        print(
            f"\nСамый агрессивный БЕЗОПАСНЫЙ порог: {best[0]}"
            f"\n  ни одного вредного отказа, при этом поймано "
            f"{best[1]} из {n_unans} неотвечаемых и {best[2]} из {n_missed} промахов."
        )
    else:
        print("\nБезопасного порога нет: любой отказ уже задевает найденные документы.")

    # Доминирующий порог: максимум пользы при минимуме вреда.
    # Строка A доминирует строку B, если пользы не меньше, а вреда не больше.
    def dominated(row) -> bool:
        return any(
            other[1] + other[2] >= row[1] + row[2] and other[3] < row[3]
            for other in rows
        )

    frontier = [r for r in rows if not dominated(r)]
    if frontier:
        pick = max(frontier, key=lambda r: (r[1] + r[2]) - r[3])
        print(
            f"\nРазумный выбор: порог {pick[0]}"
            f"\n  полезных отказов {pick[1] + pick[2]}, вредных {pick[3]}."
            f"\n  Вписать в config.py:"
            f"\n      ABSTAIN_THRESHOLDS[\"{method}\"] = {pick[0]}"
        )
        print(
            "\nПорог у каждого подхода свой: оценка берётся по разному числу\n"
            "кандидатов, поэтому чужое значение не подойдёт."
        )

    import pandas as pd

    df = pd.DataFrame(
        rows, columns=["threshold", "justified", "useful", "harmful", "missed_fabrications"]
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"abstention_{method}.csv"
    df.to_csv(out, index=False)
    print(f"\nТаблица: results/{out.name}")


if __name__ == "__main__":
    main()
