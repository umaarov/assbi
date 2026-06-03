"""LLM backends for the analytics assistant.

The assistant is *grounded*: the backend is always given the session's analytics
report as context and asked to answer only from it, so a real LLM adds natural
conversation (greetings, typos, follow-ups, summarising) without hallucinating
the numbers.

DeepSeek exposes an OpenAI-compatible Chat Completions API, so the same backend
works for any OpenAI-compatible endpoint by changing ``base_url``/``model``.

The API key is read from the environment (never hardcoded). Set it with::

    setx DEEPSEEK_API_KEY "sk-..."      # Windows (new shells)
    $env:DEEPSEEK_API_KEY = "sk-..."    # current PowerShell session
"""
from __future__ import annotations

import os


class DeepSeekBackend:
    """Calls a DeepSeek / OpenAI-compatible chat-completions endpoint.

    Implements the ``LLMBackend`` protocol (``complete(system, user) -> str``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.2,
        timeout: float = 30.0,
        env_var: str = "DEEPSEEK_API_KEY",
    ) -> None:
        self.api_key = api_key or os.environ.get(env_var)
        if not self.api_key:
            raise RuntimeError(
                f"No API key. Set ${env_var} or pass api_key=… to use the LLM chatbot."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        import requests  # lazy: keep the import out of the stdlib path

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
