"""Шаг 4. Прогнать гибрид на нескольких значениях alpha за один раз.

Запуск:  python scripts/04_sweep_alpha.py
         python scripts/04_sweep_alpha.py 0.0 0.25 0.5 0.75 1.0

Обращений к API не делает — денег не стоит.

ЗАЧЕМ

alpha в config.py задаёт, какая доля веса при слиянии достаётся векторам,
а какая словам: 1.0 — только эмбеддинги, 0.0 — только BM25.

Менять его руками и перезапускать бенчмарк по разу на значение — долго
и легко ошибиться (например, забыть сохранить файл). Скрипт строит индекс
ОДИН раз и переиспользует его для всех значений: эмбеддинги от alpha
не зависят, меняется только способ смешивания двух рейтингов.

ЧТО ЧИТАТЬ В РЕЗУЛЬТАТЕ

Значение имеет форма кривой, а не отдельная строка. Обычно кривая
немонотонна: край alpha=1.0 — чистые вектора, край alpha=0.0 — чистый
BM25, оптимум лежит между ними, ближе к тому краю, который лучше
подходит корпусу и набору вопросов.

Отдельного внимания стоит колонка semantic. Резкое падение при
уменьшении alpha означает, что лексический поиск вытесняет из выдачи
документы, найденные по смыслу.

Самопроверка: при alpha=1.0 цифры должны быть близки к подходу naive.
Сильное расхождение указывает на ошибку.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from config import QUESTIONS_FILE, RESULTS_DIR, TOP_K  # noqa: E402
from ragbench.corpus import build_chunks  # noqa: E402
from ragbench.evaluate import (  # noqa: E402
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.retrievers.hybrid import HybridRetriever  # noqa: E402

DEFAULT_ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    alphas = [float(a) for a in sys.argv[1:]] or DEFAULT_ALPHAS

    questions = [
        q
        for q in yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
        if q.get("gold_sources")          # unanswerable здесь не при чём
    ]
    chunks = build_chunks()
    print(f"Вопросов с источниками: {len(questions)}, чанков: {len(chunks)}")
    print(f"Значения alpha: {alphas}\n")

    # Индекс строится один раз и переиспользуется: alpha влияет только
    # на слияние двух рейтингов, а не на сами эмбеддинги.
    print("Строю индекс (один раз для всех значений)...")
    base = HybridRetriever(chunks, alpha=alphas[0])
    base.build()

    types = sorted({q.get("type") for q in questions})
    rows = []

    for alpha in alphas:
        base.alpha = alpha          # переключаем вес, ничего не пересчитывая

        per_type: dict[str, list[float]] = {t: [] for t in types}
        recalls, precisions, rrs = [], [], []

        for q in questions:
            r = base.search(q["question"], TOP_K)
            gold = q["gold_sources"]

            rec = recall_at_k(r.sources, gold)
            recalls.append(rec)
            precisions.append(precision_at_k(r.sources, gold))
            rrs.append(reciprocal_rank(r.sources, gold))
            per_type[q.get("type")].append(rec)

        row = {
            "alpha": alpha,
            "recall": mean(recalls),
            "precision": mean(precisions),
            "mrr": mean(rrs),
            **{t: mean(v) for t, v in per_type.items()},
        }
        rows.append(row)
        print(
            f"  alpha={alpha:<4} recall={row['recall']:.3f} "
            f"mrr={row['mrr']:.3f} semantic={row.get('semantic', float('nan')):.3f}"
        )

    # --- таблица ---
    import pandas as pd

    df = pd.DataFrame(rows).set_index("alpha").round(3)
    print("\n\nЗАВИСИМОСТЬ ОТ ALPHA (1.0 = только вектора, 0.0 = только BM25)\n")
    print(df.to_markdown())

    best = df["recall"].idxmax()
    print(f"\nЛучший recall при alpha = {best}")
    print("Чтобы закрепить, впишите в config.py:")
    print(f"    HYBRID_ALPHA = {best}")

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "alpha_sweep.csv")

    # --- график ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df["recall"], marker="o", linewidth=2, label="recall@5 (все)")
        for t in types:
            if t in df.columns:
                ax.plot(df.index, df[t], marker=".", alpha=0.75, label=t)
        ax.set_xlabel("alpha  (0 = только BM25, 1 = только эмбеддинги)")
        ax.set_ylabel("recall@5")
        ax.set_ylim(0, 1.02)
        ax.set_title("Как вес лексического поиска влияет на качество")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "alpha_sweep.png", dpi=150)
        plt.close()
        print("\nГрафик: results/alpha_sweep.png")
    except ImportError:
        pass

    print("Таблица: results/alpha_sweep.csv")


if __name__ == "__main__":
    main()
