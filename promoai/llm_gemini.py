"""Gemini provider over the plain REST ``generateContent`` endpoint.

* No SDK dependency (standard library only), so there is nothing provider-specific
  to leak into the rest of the code.
* The API key travels in the ``x-goog-api-key`` header - never in the URL - so it
  cannot appear in request-URL logs or exception messages.
* No ``responseSchema`` / JSON-mode: we only send the prompt and read text back.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .config import ConfigError, Settings
from .llm import LLMBlockedError, LLMError, LLMProvider, LLMQuotaError

log = logging.getLogger("promoai.llm.gemini")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE = {500, 502, 503, 504}

# transport(method, url, headers, body, timeout) -> (status, body_bytes); status 0 = network failure
Transport = Callable[[str, str, dict, Optional[bytes], float], "tuple[int, bytes]"]


def _urllib_transport(method: str, url: str, headers: dict, body: Optional[bytes], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        settings: Settings,
        *,
        require_model: bool = True,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if not settings.api_key:
            raise ConfigError("GEMINI_API_KEY is not set (put it in .env or the environment)")
        if require_model and not settings.model:
            raise ConfigError("LLM_MODEL is not set; run `python -m promoai.llm --list-models` and set it in .env")
        self.settings = settings
        self.model = settings.model
        self._api_key = settings.api_key
        self._transport = transport or _urllib_transport
        self._sleep, self._clock = sleep, clock
        self._last_call: Optional[float] = None

    @property
    def fingerprint(self) -> str:
        s = self.settings
        return f"gemini|{self.model}|t={s.temperature}|max={s.max_output_tokens}"

    # ------------------------------------------------------------------ helpers
    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "<redacted>") if self._api_key else text

    def _throttle(self) -> None:
        gap = self.settings.min_interval_seconds
        if self._last_call is not None and gap > 0:
            wait = gap - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    @staticmethod
    def _error_info(body: bytes) -> tuple[str, str, Optional[float]]:
        """-> (message, raw_lowercase_text, retry_delay_seconds)"""
        raw = body.decode("utf-8", "replace")
        message, delay = raw[:300], None
        try:
            err = json.loads(raw).get("error", {})
            message = str(err.get("message", message))[:300]
            for d in err.get("details", []) or []:
                m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
                if m:
                    delay = float(m.group(1))
        except (ValueError, AttributeError):
            pass
        return message, raw.lower(), delay

    def _request(self, method: str, url: str, payload: Optional[dict]) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        retries = self.settings.max_transport_retries
        for attempt in range(retries + 1):
            self._throttle()
            status, data = self._transport(method, url, headers, body, self.settings.timeout_seconds)
            if status == 200:
                try:
                    return json.loads(data)
                except ValueError:
                    self.stats.record_failure("bad_response_body")
                    raise LLMError("Gemini returned a non-JSON HTTP body") from None
            message, raw, delay = self._error_info(data)
            message = self._redact(message)
            if status == 429 and ("perday" in raw or "per day" in raw):
                self.stats.record_failure("quota_daily", "429")
                raise LLMQuotaError(f"daily quota exhausted for this model ({message}). Try a different model or wait for reset.")
            retryable = status == 429 or status in RETRYABLE or status == 0
            if not retryable:
                self.stats.record_failure(f"http_{status}", message[:80])
                hint = " (check LLM_MODEL with --list-models)" if status == 404 else ""
                raise LLMError(f"Gemini HTTP {status}: {message}{hint}")
            kind = "network" if status == 0 else f"http_{status}"
            self.stats.record_failure(kind, "will retry" if attempt < retries else "giving up")
            if attempt == retries:
                raise LLMError(f"Gemini {kind} persisted after {retries} retries: {message}")
            wait = min(delay + 1 if delay else 2.0 * (2 ** attempt), 120.0)
            log.warning("gemini %s; retry %d/%d in %.0fs", kind, attempt + 1, retries, wait)
            self._sleep(wait)
        raise LLMError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ public
    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.settings.temperature,
                "maxOutputTokens": self.settings.max_output_tokens,
            },
        }
        data = self._request("POST", f"{BASE_URL}/models/{self.model}:generateContent", payload)

        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            self.stats.record_failure("blocked", str(feedback["blockReason"]))
            raise LLMBlockedError(f"prompt blocked by provider: {feedback['blockReason']}")
        cands = data.get("candidates") or []
        if not cands:
            self.stats.record_failure("no_candidates")
            raise LLMError("Gemini returned no candidates")
        cand = cands[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = cand.get("finishReason", "")
        if not text.strip():
            self.stats.record_failure("empty_text", str(finish))
            if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"}:
                raise LLMBlockedError(f"reply blocked by provider: {finish}")
            raise LLMError(f"Gemini returned empty text (finishReason={finish or 'unknown'})")
        if finish == "MAX_TOKENS":
            # Not fatal here: truncated JSON will fail validation and trigger the repair retry.
            self.stats.record_failure("truncated", "MAX_TOKENS")

        u = data.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": int(u.get("promptTokenCount", 0)),
            "output_tokens": int(u.get("candidatesTokenCount", 0)),
            "thinking_tokens": int(u.get("thoughtsTokenCount", 0)),
            "total_tokens": int(u.get("totalTokenCount", 0)),
        }
        self.last_usage = usage
        self.stats.record_call(usage)
        return text

    def list_models(self) -> list[str]:
        """Model IDs (without the ``models/`` prefix) that support generateContent."""
        names: list[str] = []
        url = f"{BASE_URL}/models?pageSize=100"
        data = self._request("GET", url, None)
        for m in data.get("models", []):
            if "generateContent" in (m.get("supportedGenerationMethods") or []):
                names.append(str(m.get("name", "")).removeprefix("models/"))
        return sorted(n for n in names if n)
