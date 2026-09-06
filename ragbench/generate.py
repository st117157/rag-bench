"""Генерация ответа по найденному контексту.

Этот файл одинаковый для всех пяти подходов — и это принципиально.
Меняется только то, ЧТО попало в контекст. Если менять ещё и промпт,
сравнение перестанет что-либо значить.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragbench.corpus import Chunk
from ragbench.llm import complete

# Промпт на английском — на языке корпуса и вопросов. Смешивать языки
# здесь вредно: модель начинает переводить вместо того, чтобы отвечать,
# и в ответ протекают знания из обучения вместо содержимого фрагментов.
ANSWER_SYSTEM = """You answer questions about the FastAPI documentation.

Rules:
- answer ONLY from the provided fragments, never from your own knowledge;
- if the fragments do not contain the answer, reply exactly: NO DATA
- keep it short, one to three sentences;
- end with the sources in the form [source: filename.md]."""

NO_DATA = "NO DATA"


@dataclass
class Answer:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def abstained(self) -> bool:
        """Модель честно отказалась отвечать."""
        return NO_DATA in self.text.upper()


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    if not chunks:
        context = "(no fragments found)"
    else:
        context = "\n\n---\n\n".join(c.with_header() for c in chunks)
    return f"DOCUMENTATION FRAGMENTS:\n\n{context}\n\nQUESTION: {question}"


def answer(
    question: str, chunks: list[Chunk], abstain_hint: bool | None = None
) -> Answer:
    """Сгенерировать ответ по найденному контексту.

    abstain_hint приходит от ретривера с порогом отказа (см. reranked.py).
    Если он говорит «ответа в корпусе нет», обращаться к модели незачем:
    мы и так знаем ответ, и вызов только потратил бы деньги на то,
    чтобы дать модели шанс что-нибудь выдумать.
    """
    if abstain_hint is True:
        return Answer(text=NO_DATA, input_tokens=0, output_tokens=0)

    resp = complete(build_prompt(question, chunks), system=ANSWER_SYSTEM)
    return Answer(
        text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# Замечание по результатам: эта инструкция оказалась СЛИШКОМ строгой.
# Три четверти неверных ответов в прогоне — отказы «NO DATA», причём
# в значительной части случаев нужный документ был в контексте.
# Что стоит попробовать дальше:
#   1) смягчить формулировку отказа — отвечать, если фрагменты покрывают
#      вопрос хотя бы частично, и отказываться только при полном
#      отсутствии связи;
#   2) велеть сначала выписать релевантные цитаты и только потом
#      отвечать — тогда решение об отказе принимается по наличию цитат,
#      а не по общему впечатлению;
#   3) поднять TOP_K: отказы напрямую зависят от объёма контекста,
#      у подходов с десятками документов их нет вовсе.
