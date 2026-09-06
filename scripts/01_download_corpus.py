"""Шаг 1. Скачать корпус — документацию FastAPI.

Запуск:  python scripts/01_download_corpus.py

Работает двумя способами:
  1) через git clone, если git установлен — быстрее, качается меньше;
  2) если git нет — скачивает zip-архив репозитория по https,
     ничего кроме стандартной библиотеки Python для этого не нужно.

Итог одинаковый: markdown-файлы английской документации
складываются в data/corpus/.

Почему именно FastAPI: около 150 .md файлов, чистая структура,
осмысленные заголовки. Идеальный размер для пет-проекта —
достаточно большой, чтобы поиск был нетривиальным, достаточно
маленький, чтобы целиком влезть в длинный контекст.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CORPUS_DIR, ROOT  # noqa: E402

REPO = "https://github.com/fastapi/fastapi.git"
ZIP_URL = "https://codeload.github.com/fastapi/fastapi/zip/refs/heads/master"
DOCS_SUBPATH = "docs/en/docs"
TMP_CLONE = ROOT / "data" / "_tmp_clone"

# Файлы, которые портят бенчмарк, — выкидываем сразу.
# release-notes.md весит 700 КБ, это почти половина всего корпуса:
# один этот файл перекосил бы и статистику чанков, и бюджет
# long-context ветки, а полезных ответов в нём почти нет.
# _llm-test.md — служебный файл-пустышка из репозитория FastAPI.
EXCLUDE = {"release-notes.md", "_llm-test.md"}


def have_git() -> bool:
    """Проверить, доступна ли команда git."""
    return shutil.which("git") is not None


def save(rel_path: str, data: bytes) -> bool:
    """Сохранить один файл документации в корпус.

    Имена уплощаются: tutorial/security/oauth2-jwt.md
    -> tutorial__security__oauth2-jwt.md
    Так grep-ветка сможет использовать путь как подсказку.
    Возвращает False, если файл в списке исключений.
    """
    name = rel_path.split("/")[-1]
    if name in EXCLUDE:
        return False
    flat = rel_path.replace("/", "__")
    (CORPUS_DIR / flat).write_bytes(data)
    return True


def fetch_via_git() -> tuple[int, int]:
    if TMP_CLONE.exists():
        shutil.rmtree(TMP_CLONE)

    print(f"Клонирую {REPO} (только последний коммит)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", REPO, str(TMP_CLONE)],
        check=True,
    )

    src = TMP_CLONE / DOCS_SUBPATH
    if not src.exists():
        sys.exit(
            f"Не нашёл {DOCS_SUBPATH} в репозитории. "
            "Возможно, структура поменялась — загляните в клон вручную."
        )

    saved = skipped = 0
    for md in src.rglob("*.md"):
        rel = str(md.relative_to(src)).replace("\\", "/")
        if save(rel, md.read_bytes()):
            saved += 1
        else:
            skipped += 1

    shutil.rmtree(TMP_CLONE)
    return saved, skipped


def fetch_via_zip() -> tuple[int, int]:
    print("git не найден — качаю zip-архив репозитория...")
    print(f"  {ZIP_URL}")
    print("  Это ~30 МБ, займёт от нескольких секунд до пары минут.")

    # User-Agent обязателен: без него GitHub отвечает 403.
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "rag-bench/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            blob = resp.read()
    except Exception as exc:
        sys.exit(
            f"Не удалось скачать архив: {exc}\n\n"
            "Что попробовать:\n"
            "  - проверьте интернет и отключите VPN, если он мешает GitHub;\n"
            f"  - откройте {ZIP_URL} в браузере, скачайте вручную,\n"
            "    распакуйте и скопируйте файлы из папки docs/en/docs\n"
            "    в data/corpus (имена при этом уплощать не обязательно)."
        )

    print(f"  Скачано {len(blob) / 1_048_576:.1f} МБ, распаковываю...")

    saved = skipped = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        marker = f"/{DOCS_SUBPATH}/"
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".md"):
                continue
            if marker not in info.filename:
                continue
            # В архиве путь вида fastapi-master/docs/en/docs/tutorial/body.md
            rel = info.filename.split(marker, 1)[1]
            if save(rel, z.read(info)):
                saved += 1
            else:
                skipped += 1
    return saved, skipped


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    saved, skipped = fetch_via_git() if have_git() else fetch_via_zip()

    print(f"Готово: {saved} файлов в {CORPUS_DIR} (пропущено по фильтру: {skipped})")

    if saved < 50:
        print(
            "ВНИМАНИЕ: файлов подозрительно мало. "
            "Проверьте DOCS_SUBPATH — структура репозитория могла поменяться."
        )


if __name__ == "__main__":
    main()
