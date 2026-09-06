"""Главный скрипт. Прогоняет все подходы по всем вопросам.

Запуск:
    python eval/run_eval.py                       # всё целиком (платно)
    python eval/run_eval.py --methods naive hybrid
    python eval/run_eval.py --limit 5             # быстрая проверка
    python eval/run_eval.py --no-judge            # без судьи, ответы генерируются
    python eval/run_eval.py --retrieval-only      # СОВСЕМ БЕСПЛАТНО

Про --retrieval-only. В этом режиме считаются только метрики поиска:
recall@5 и MRR. Ни одного обращения к API не происходит, деньги не тратятся.

И это не урезанная версия бенчмарка, а его основная часть. Вопрос «какой
способ поиска лучше находит нужный документ» решается целиком здесь.
Генерация ответов и LLM-судья отвечают на другой вопрос: конвертируется ли
хороший поиск в хороший ответ.

Исключение — подход grep: ему LLM нужна для придумывания поисковых
запросов, без API он не работает и в этом режиме пропускается.

Полный платный прогон = 50 вопросов x 7 подходов x 2 вызова LLM,
около 700 запросов и примерно $1 на gpt-4o-mini. Основная часть
стоимости приходится на longctx, отправляющий ~100k токенов на вопрос.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from config import (  # noqa: E402
    PRICE_PER_1M_INPUT,
    PRICE_PER_1M_OUTPUT,
    QUESTIONS_FILE,
    RESULTS_DIR,
    TOP_K,
)
from ragbench.corpus import build_chunks  # noqa: E402
from ragbench.evaluate import (  # noqa: E402
    abstention_correct,
    judge_answer,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.generate import answer  # noqa: E402
from ragbench.retrievers import NEEDS_LLM, REGISTRY  # noqa: E402


def load_questions(limit: int | None = None) -> list[dict]:
    data = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def run_one_method(
    name: str,
    chunks,
    questions: list[dict],
    use_judge: bool,
    retrieval_only: bool = False,
) -> list[dict]:
    print(f"\n{'=' * 60}\nПодход: {name}\n{'=' * 60}")

    if retrieval_only and name in NEEDS_LLM:
        print(
            f"  Пропускаю: подходу '{name}' нужна LLM для самого поиска, "
            "без API он не работает."
        )
        return []

    retriever = REGISTRY[name](chunks)

    t0 = time.perf_counter()
    retriever.build()
    build_time = time.perf_counter() - t0
    print(f"Индексация: {build_time:.1f} с")

    rows: list[dict] = []
    failures = 0
    for q in questions:
        gold = q.get("gold_sources", []) or []
        is_unanswerable = q.get("type") == "unanswerable"

        try:
            rows.append(
                _run_one_question(
                    retriever, name, q, gold, is_unanswerable,
                    use_judge, retrieval_only,
                )
            )
        except Exception as exc:
            # Сетевой обрыв на одном вопросе не должен убивать прогон,
            # который идёт двадцать минут и стоит денег. Записываем,
            # что случилось, и идём дальше.
            failures += 1
            print(f"  [x] {q['id']}  ОШИБКА: {type(exc).__name__}: {exc}")
            rows.append({
                "method": name, "qid": q["id"], "type": q.get("type", "?"),
                "error": f"{type(exc).__name__}: {exc}",
                "recall": float("nan"), "precision": float("nan"),
                "mrr": float("nan"), "n_sources": 0, "best_score": None,
                "correct": None, "abstained_ok": None, "latency_s": 0.0,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "retrieved": [], "trace": [], "answer": None,
            })
            if failures >= 5:
                print(
                    f"  Пять ошибок подряд на подходе '{name}' — "
                    "похоже, проблема системная. Прекращаю этот подход."
                )
                break
            continue

        mark = "+" if rows[-1]["recall"] == 1.0 else ("-" if gold else "~")
        print(f"  [{mark}] {q['id']}  {rows[-1]['latency_s']:.2f}с  "
              f"${rows[-1]['cost_usd']:.5f}")

    for row in rows:
        row["build_time_s"] = round(build_time, 1)

    # Промежуточное сохранение: подход отработал — результат на диске.
    # Без этого падение на последнем подходе уничтожает всю работу,
    # включая уже оплаченные запросы.
    RESULTS_DIR.mkdir(exist_ok=True)
    part = RESULTS_DIR / f"partial_{name}.json"
    part.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  сохранено: results/{part.name}")

    return rows


def _run_one_question(
    retriever, name: str, q: dict, gold: list[str],
    is_unanswerable: bool, use_judge: bool, retrieval_only: bool,
) -> dict:
    """Один вопрос через один подход. Вынесено, чтобы обернуть в try."""
    r = retriever.timed_search(q["question"], TOP_K)

    # В режиме retrieval_only генерации нет вообще: считаем только
    # метрики поиска. Они и есть суть бенчмарка, а стоят ноль.
    a = (
        None
        if retrieval_only
        else answer(q["question"], r.chunks, abstain_hint=r.abstain_hint)
    )

    in_tok = r.extra_input_tokens + (a.input_tokens if a else 0)
    out_tok = r.extra_output_tokens + (a.output_tokens if a else 0)

    correct = None
    if a is not None and use_judge and not is_unanswerable:
        j = judge_answer(q["question"], q["gold_answer"], a.text)
        correct = j.correct
        # Токены судьи в стоимость системы НЕ включаем: это расход
        # на оценку, а не на работу. Но упомяните его в README.

    return {
        "method": name,
        "qid": q["id"],
        "type": q.get("type", "?"),
        "recall": recall_at_k(r.sources, gold),
        "precision": precision_at_k(r.sources, gold),
        "mrr": reciprocal_rank(r.sources, gold),
        "n_sources": len(r.sources),
        "best_score": r.best_score,
        "correct": correct,
        "abstained_ok": (
            abstention_correct(is_unanswerable, r.abstain_hint)
            if a is None and r.abstain_hint is not None
            else None
            if a is None
            else abstention_correct(is_unanswerable, a.abstained)
        ),
        "latency_s": round(r.latency_s, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(
            in_tok / 1e6 * PRICE_PER_1M_INPUT + out_tok / 1e6 * PRICE_PER_1M_OUTPUT,
            6,
        ),
        "retrieved": r.sources[:TOP_K],
        "trace": r.trace,
        "answer": a.text if a else None,
    }


def summarize(rows: list[dict]) -> None:
    import pandas as pd

    if not rows:
        print("\nНечего показывать: ни один подход не отработал.")
        return

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("method")
        .agg(
            recall_at_5=("recall", "mean"),
            precision=("precision", "mean"),
            mrr=("mrr", "mean"),
            docs_sent=("n_sources", "mean"),
            correct=("correct", "mean"),
            abstain_ok=("abstained_ok", "mean"),
            latency_s=("latency_s", "mean"),
            cost_per_q=("cost_usd", "mean"),
            build_s=("build_time_s", "first"),
        )
    )
    # Стоимость одного запроса — величина порядка тысячных долей доллара,
    # при round(3) она превратится в ноль. Поэтому колонки округляем
    # по-разному: качество до 3 знаков, деньги — до 5.
    cost_cols = ["cost_per_q"]
    summary[cost_cols] = summary[cost_cols].round(5)
    summary[[c for c in summary.columns if c not in cost_cols]] = summary[
        [c for c in summary.columns if c not in cost_cols]
    ].round(3)
    # Плюс отдельная колонка «сколько стоит 1000 запросов» —
    # в таком виде цифры наглядны и их не стыдно положить в README.
    summary["cost_per_1k_q"] = (summary["cost_per_q"] * 1000).round(2)

    # Колонки, где всё пусто (например, correct в режиме retrieval-only),
    # только загромождают таблицу — убираем их.
    summary = summary.dropna(axis=1, how="all")

    print("\n\nИТОГОВАЯ ТАБЛИЦА\n")
    print(summary.to_markdown())

    print("\n\nRECALL@5 ПО ТИПАМ ВОПРОСОВ\n")
    # Вот это — самая интересная таблица во всём проекте.
    # Здесь должно быть видно, что grep проваливается на semantic,
    # а на lexical идёт вровень с векторами или лучше.
    by_type = df.pivot_table(
        index="method", columns="type", values="recall", aggfunc="mean"
    ).round(3)
    print(by_type.to_markdown())

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"raw_{stamp}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary.to_csv(RESULTS_DIR / f"summary_{stamp}.csv")
    print(f"\nСырые результаты: results/raw_{stamp}.json")

    make_plots(df, summary, by_type, stamp)


def make_plots(df, summary, by_type, stamp: str) -> None:
    """Два графика для README.

    Первый — recall по типам вопросов. Он показывает главное: у каждого
    подхода свой профиль сильных и слабых мест, и усреднённая цифра
    этот профиль скрывает.

    Второй — стоимость против качества. Точка в левом верхнем углу
    означает «дёшево и хорошо», в правом нижнем — «дорого и плохо».
    Обычно оказывается, что самый дорогой подход не самый лучший,
    и увидеть это глазами убедительнее, чем прочитать в таблице.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")     # без окна, просто пишем файлы
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — графики пропущены")
        return

    # --- график 1: recall по типам ---
    if not by_type.empty:
        ax = by_type.plot(kind="bar", figsize=(9, 5), width=0.8)
        ax.set_ylabel("recall@5")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_title("Что каждый подход находит, по типам вопросов")
        ax.legend(title="тип вопроса", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"recall_by_type_{stamp}.png", dpi=150)
        plt.close()

    # --- график 2: цена против качества ---
    if summary["cost_per_q"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, row in summary.iterrows():
            ax.scatter(row["cost_per_1k_q"], row["recall_at_5"], s=120)
            ax.annotate(
                method,
                (row["cost_per_1k_q"], row["recall_at_5"]),
                textcoords="offset points",
                xytext=(8, 4),
            )
        ax.set_xlabel("стоимость 1000 запросов, USD")
        ax.set_ylabel("recall@5")
        ax.set_title("Цена против качества: левый верхний угол лучше")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"cost_vs_quality_{stamp}.png", dpi=150)
        plt.close()

    print(f"Графики: results/recall_by_type_{stamp}.png, "
          f"results/cost_vs_quality_{stamp}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="не вызывать LLM-судью (ответы всё равно генерируются)",
    )
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="считать ТОЛЬКО метрики поиска, без единого обращения к API. "
        "Бесплатно. Подход grep в этом режиме пропускается — ему нужна LLM.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="не пересчитывать подходы, для которых уже есть "
        "results/partial_<подход>.json. Нужно после обрыва: доплачивать "
        "за то, что уже посчитано, незачем.",
    )
    args = p.parse_args()

    questions = load_questions(args.limit)
    print(f"Вопросов: {len(questions)}, подходов: {len(args.methods)}")
    if args.retrieval_only:
        print("Режим: только поиск, обращений к API не будет (бесплатно)")

    chunks = build_chunks()
    print(f"Чанков в корпусе: {len(chunks)}")

    all_rows: list[dict] = []
    try:
        for m in args.methods:
            part = RESULTS_DIR / f"partial_{m}.json"
            if args.resume and part.exists():
                saved = json.loads(part.read_text(encoding="utf-8"))
                print(f"\n{'=' * 60}\nПодход: {m}\n{'=' * 60}")
                print(f"  Беру готовое из results/{part.name} ({len(saved)} строк)")
                all_rows += saved
                continue

            all_rows += run_one_method(
                m,
                chunks,
                questions,
                use_judge=not args.no_judge,
                retrieval_only=args.retrieval_only,
            )
    except KeyboardInterrupt:
        print("\nПрервано вручную. Свожу то, что успело посчитаться.")
    finally:
        # Таблица строится по всему, что успело накопиться, даже если
        # прогон оборвался. Оплаченные запросы не должны пропадать зря.
        summarize(all_rows)

        errors = [r for r in all_rows if r.get("error")]
        if errors:
            print(f"\nОшибок при обращении к API: {len(errors)}")
            for r in errors[:5]:
                print(f"  {r['method']} / {r['qid']}: {r['error']}")
            print(
                "\nСтроки с ошибками попали в таблицу как пустые и занижают\n"
                "метрики. Прогоните эти подходы заново, когда сеть стабильна."
            )


if __name__ == "__main__":
    main()
