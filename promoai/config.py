"""Configuration: real environment variables win over a local ``.env`` file.

No third-party dependency: the ``.env`` format we need is just ``KEY=VALUE``.
Secrets are never printed: ``Settings.__repr__`` hides the API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


class ConfigError(RuntimeError):
    pass


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal .env file.  Missing file -> empty dict."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:  # strip trailing comment on unquoted values
            val = val.split(" #", 1)[0].strip()
        values[key] = val
    return values


@dataclass(frozen=True)
class Settings:
    provider: str = "gemini"
    model: str = ""
    api_key: str = field(default="", repr=False)   # repr=False: never leaks into logs
    temperature: float = 0.4              # proposer / story model (Phase 2)
    judge_temperature: float = 0.0        # semantic judge (Phase 3); separate so judging can be made as repeatable as possible
    judge_votes: int = 1                  # independent judge samples per clip; >1 enables majority voting (off by default)
    judge_votes_set: bool = False         # True when LLM_JUDGE_VOTES was given explicitly (run.py then honours it over its own default)
    max_output_tokens: int = 8192
    min_interval_seconds: float = 6.0     # ~10 requests/minute; raise if you hit 429s
    timeout_seconds: float = 180.0
    max_transport_retries: int = 3
    cache_dir: Path = Path("data/cache/llm")

    def __repr__(self) -> str:  # explicit, in case someone prints the object
        key = "set" if self.api_key else "missing"
        return f"Settings(provider={self.provider!r}, model={self.model!r}, api_key=<{key}>, cache_dir={str(self.cache_dir)!r})"


def load_settings(env_file: Path = Path(".env"), environ: Optional[Mapping[str, str]] = None) -> Settings:
    env = dict(load_dotenv(env_file))
    env.update(os.environ if environ is None else environ)  # real environment overrides .env

    def f(name: str, default: float) -> float:
        try:
            return float(env[name]) if env.get(name, "") != "" else default
        except ValueError:
            raise ConfigError(f"{name} must be a number") from None

    provider = (env.get("LLM_PROVIDER") or "gemini").strip().lower()
    key_var = f"{provider.upper()}_API_KEY"
    votes = f("LLM_JUDGE_VOTES", 1)
    if votes < 1 or votes != int(votes):
        raise ConfigError("LLM_JUDGE_VOTES must be a whole number >= 1")
    return Settings(
        provider=provider,
        model=(env.get("LLM_MODEL") or "").strip(),
        api_key=(env.get(key_var) or "").strip(),
        temperature=f("LLM_TEMPERATURE", 0.4),
        judge_temperature=f("LLM_JUDGE_TEMPERATURE", 0.0),
        judge_votes=int(votes),
        judge_votes_set=env.get("LLM_JUDGE_VOTES", "") != "",
        max_output_tokens=int(f("LLM_MAX_OUTPUT_TOKENS", 8192)),
        min_interval_seconds=f("LLM_MIN_INTERVAL_SECONDS", 6.0),
        timeout_seconds=f("LLM_TIMEOUT_SECONDS", 180.0),
        max_transport_retries=int(f("LLM_MAX_TRANSPORT_RETRIES", 3)),
        cache_dir=Path(env.get("LLM_CACHE_DIR") or "data/cache/llm"),
    )
