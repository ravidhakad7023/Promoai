"""Provider-independent LLM access.

The rest of the project only knows this interface::

    provider.generate(prompt: str) -> str

Everything else lives around it:

* :func:`create_provider`   - build the provider named in configuration
* :class:`CachedProvider`   - disk cache so a repeated prompt never costs an API call
* :func:`generate_validated`- ask for JSON, validate with Pydantic, retry once if invalid
* :class:`FakeProvider`     - scripted provider for tests / offline runs
* :class:`UsageStats`       - real call / token / failure accounting

Design rules
------------
* No provider-specific "structured output" features are used.  We ask for JSON in
  the prompt and validate it ourselves, so swapping Gemini for another model only
  means adding one small class with a ``generate`` method.
* Prompts and responses contain confidential screenplay text: they are **never
  logged**.  The cache stores responses (not prompts) under a hash key, in a
  git-ignored folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .config import ConfigError, Settings, load_settings

log = logging.getLogger("promoai.llm")

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- errors
class LLMError(RuntimeError):
    """Any failure talking to a model."""


class LLMQuotaError(LLMError):
    """Daily / hard quota exhausted: retrying today is pointless."""


class LLMBlockedError(LLMError):
    """The provider refused to answer (e.g. safety filter)."""


class LLMResponseError(LLMError):
    """The model answered, but not with valid JSON for the requested schema."""


# --------------------------------------------------------------------------- usage
@dataclass
class UsageStats:
    api_calls: int = 0            # real network calls that returned a model answer
    cache_hits: int = 0
    json_retries: int = 0         # repair attempts after invalid JSON
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    tokens_saved_by_cache: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, usage: dict[str, int]) -> None:
        self.api_calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.thinking_tokens += usage.get("thinking_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    def record_failure(self, kind: str, detail: str = "") -> None:
        # detail must never contain prompt/response text - callers pass short, generic strings
        self.failures.append({"kind": kind, "detail": detail[:200]})

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "json_retries": self.json_retries,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "tokens_saved_by_cache": self.tokens_saved_by_cache,
            "failures": list(self.failures),
        }


# --------------------------------------------------------------------------- interface
class LLMProvider(ABC):
    """Minimal provider contract."""

    name: str = "base"
    model: str = ""
    on_invalid = None      # optional callable(prompt=, attempt=, raw=, error=) told about every unusable reply

    def __init__(self) -> None:
        self.stats = UsageStats()
        self.last_usage: dict[str, int] = {}

    @property
    def fingerprint(self) -> str:
        """Everything that changes the answer; part of the cache key."""
        return f"{self.name}|{self.model}"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send ``prompt``, return the model's text."""

    # Cache hooks - no-ops unless wrapped by CachedProvider.
    def invalidate(self, prompt: str) -> None:  # pragma: no cover - trivial
        pass

    def remember(self, prompt: str, text: str) -> None:  # pragma: no cover - trivial
        pass


class FakeProvider(LLMProvider):
    """Scripted provider for tests and offline dry runs.  Makes no network calls.

    ``script`` is a list of replies (strings, or callables ``prompt -> str``) used in
    order; the last one repeats.  Prompts are kept in memory only (``self.prompts``)."""

    name = "fake"

    def __init__(self, script: Union[Sequence[Union[str, Callable[[str], str]]], Callable[[str], str]], model: str = "fake-1") -> None:
        super().__init__()
        self.model = model
        self._script = script
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        i = len(self.prompts)
        self.prompts.append(prompt)
        item = self._script
        if not callable(item):
            seq = list(item)
            item = seq[min(i, len(seq) - 1)]
        text = item(prompt) if callable(item) else item
        usage = {"prompt_tokens": len(prompt) // 4, "output_tokens": len(text) // 4,
                 "total_tokens": (len(prompt) + len(text)) // 4}
        self.last_usage = usage
        self.stats.record_call(usage)
        return text


# --------------------------------------------------------------------------- cache
class CachedProvider(LLMProvider):
    """Disk cache around any provider.

    Key = sha256(provider|model|settings|prompt).  Only the *response* and its token
    usage are stored (never the prompt).  Corrupt cache files count as a miss."""

    def __init__(self, inner: LLMProvider, cache_dir: Path, salt: str = "") -> None:
        super().__init__()
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.salt = salt
        self.name, self.model = inner.name, inner.model
        self.stats = inner.stats          # one shared set of counters

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def salted(self, salt: str) -> "CachedProvider":
        """Same provider, same cache folder, but a separate set of cache keys.

        Needed when the SAME prompt must be answered several times independently (repeat runs, judge
        votes): without a salt the cache would replay the first answer and the repeats would be fake."""
        other = CachedProvider(self.inner, self.cache_dir, salt)
        other.on_invalid = self.on_invalid          # keep saving unusable raw replies from salted providers too
        return other

    def _key(self, prompt: str) -> str:
        # An empty salt keeps the original key format, so entries written before salting existed still match.
        base = f"{self.fingerprint}\0{prompt}" if not self.salt else f"{self.fingerprint}\0{self.salt}\0{prompt}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read(self, key: str) -> Optional[dict]:
        p = self._path(key)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data.get("text"), str) else None
        except (OSError, ValueError, AttributeError):
            return None

    def _write(self, key: str, text: str, usage: dict[str, int]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"text": text, "usage": usage, "provider": self.name, "model": self.model,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, p)       # atomic: a crash never leaves a half-written entry
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def generate(self, prompt: str) -> str:
        key = self._key(prompt)
        hit = self._read(key)
        if hit is not None:
            self.stats.cache_hits += 1
            self.stats.tokens_saved_by_cache += int(hit.get("usage", {}).get("total_tokens", 0))
            self.last_usage = {}
            log.info("cache hit %s", key[:10])
            return hit["text"]
        text = self.inner.generate(prompt)
        self.last_usage = self.inner.last_usage
        self._write(key, text, self.inner.last_usage)
        return text

    def invalidate(self, prompt: str) -> None:
        try:
            self._path(self._key(prompt)).unlink()
        except OSError:
            pass

    def remember(self, prompt: str, text: str) -> None:
        self._write(self._key(prompt), text, {})

    def __getattr__(self, item: str) -> Any:   # delegate e.g. list_models()
        if item in {"inner", "stats", "cache_dir"}:
            raise AttributeError(item)
        return getattr(self.inner, item)


# --------------------------------------------------------------------------- JSON handling
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n?(.*?)\n?```\s*$", re.S)


def extract_json(text: str) -> Any:
    """Parse a JSON value from model output.

    Tolerates a Markdown code fence or chatter around a single JSON object, since
    we deliberately do not use any provider-side JSON mode."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("no JSON object found in model output")


def _error_summary(exc: Exception, limit: int = 15) -> str:
    if isinstance(exc, ValidationError):
        rows = [f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()[:limit]]
        return "The JSON did not match the required schema:\n" + "\n".join(rows)
    return f"The reply was not valid JSON ({type(exc).__name__}: {str(exc)[:200]})."


REPAIR_SUFFIX = (
    "\n\n=====\nYour previous reply could not be used.\n{problem}\n\n"
    "Previous reply (may be truncated):\n{previous}\n\n"
    "Return ONLY the corrected, complete JSON object. No explanations, no code fences."
)


def _report_invalid(provider: LLMProvider, prompt: str, attempt: int, raw: str, exc: Exception) -> None:
    """Hand an unusable reply to ``provider.on_invalid`` (if set) so a run can keep the raw text + error."""
    sink = getattr(provider, "on_invalid", None)
    if sink is not None:
        try:
            sink(prompt=prompt, attempt=attempt, raw=raw, error=_error_summary(exc))
        except Exception:  # a broken sink must never break the pipeline
            log.warning("on_invalid sink failed")


def generate_validated(provider: LLMProvider, prompt: str, model_cls: type[T]) -> tuple[T, int]:
    """Call the model, parse + validate against ``model_cls``.  One repair retry.

    Returns ``(object, attempts)`` where attempts is 1 or 2.  Raises
    :class:`LLMResponseError` if the second reply is also unusable.  Transport errors
    (:class:`LLMError`) propagate unchanged.  Every unusable reply is passed to ``provider.on_invalid``."""
    text = provider.generate(prompt)
    try:
        return model_cls.model_validate(extract_json(text)), 1
    except (ValueError, ValidationError) as first:
        _report_invalid(provider, prompt, 1, text, first)
        provider.stats.json_retries += 1
        provider.stats.record_failure("invalid_json", type(first).__name__)
        log.warning("model reply failed validation (%s); retrying once", type(first).__name__)
        provider.invalidate(prompt)       # don't let a bad reply live in the cache
        repair = prompt + REPAIR_SUFFIX.format(problem=_error_summary(first), previous=text[:6000])
        text2 = provider.generate(repair)
        try:
            obj = model_cls.model_validate(extract_json(text2))
        except (ValueError, ValidationError) as second:
            _report_invalid(provider, prompt, 2, text2, second)
            provider.stats.record_failure("invalid_json_after_retry", type(second).__name__)
            provider.invalidate(repair)
            raise LLMResponseError(f"model reply still invalid after retry: {_error_summary(second)}") from second
        provider.remember(prompt, text2)   # next run: one cache hit, zero API calls
        return obj, 2


# --------------------------------------------------------------------------- factory + CLI
def create_provider(settings: Optional[Settings] = None, *, cache: bool = True) -> LLMProvider:
    settings = settings or load_settings()
    if settings.provider == "gemini":
        from .llm_gemini import GeminiProvider

        inner: LLMProvider = GeminiProvider(settings)
    else:
        raise ConfigError(f"unknown LLM_PROVIDER {settings.provider!r} (available: gemini)")
    return CachedProvider(inner, settings.cache_dir) if cache else inner


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LLM provider utilities")
    ap.add_argument("--list-models", action="store_true", help="list models your key can call generateContent on")
    args = ap.parse_args(argv)
    if not args.list_models:
        ap.print_help()
        return 0
    try:
        settings = load_settings()
        if settings.provider != "gemini":
            raise ConfigError("--list-models is implemented for the gemini provider")
        from .llm_gemini import GeminiProvider

        names = GeminiProvider(settings, require_model=False).list_models()
    except (ConfigError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(names) or "(no models returned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
