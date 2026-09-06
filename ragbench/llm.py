"""Тонкая обёртка над LLM API.

Одна точка входа для всех вызовов модели — чтобы:
  1) везде считались токены (без этого не будет колонки «стоимость»);
  2) можно было поменять провайдера в одном месте.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config import LLM_BASE_URL, LLM_MODEL, PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * PRICE_PER_1M_INPUT
            + self.output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT
        )


_client = None

# Полный прогон — это сотни запросов подряд минут на двадцать. За такое время
# сетевой обрыв почти неизбежен: разорванное соединение, таймаут рукопожатия
# TLS, кратковременный 429. По умолчанию клиент повторяет попытку дважды,
# для длинных платных прогонов этого мало.
MAX_RETRIES = 6
REQUEST_TIMEOUT_S = 90.0


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "Не найден OPENAI_API_KEY. Скопируйте .env.example в .env "
                "и вставьте ключ."
            )
        kwargs = {"max_retries": MAX_RETRIES, "timeout": REQUEST_TIMEOUT_S}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _client = OpenAI(**kwargs)
    return _client


def complete(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> LLMResponse:
    """Один вызов модели. temperature=0 — чтобы бенчмарк был воспроизводим."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return LLMResponse(
        text=resp.choices[0].message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )


_encoder = None
_encoder_failed = False


def count_tokens(text: str) -> int:
    """Оценка числа токенов без вызова API — нужна для long-context ветки.

    tiktoken при первом запуске скачивает таблицу токенов из интернета.
    За корпоративным прокси это иногда не работает, поэтому предусмотрен
    грубый запасной вариант: ~4 символа на токен для английского текста.
    Для бюджетирования контекста такой точности достаточно.
    """
    global _encoder, _encoder_failed

    if _encoder is None and not _encoder_failed:
        try:
            import tiktoken

            try:
                _encoder = tiktoken.encoding_for_model(LLM_MODEL)
            except KeyError:
                _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # нет сети, нет пакета — не падаем
            _encoder_failed = True
            print(f"[llm] tiktoken недоступен ({exc.__class__.__name__}), "
                  "считаю токены приблизительно")

    if _encoder is not None:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4)
