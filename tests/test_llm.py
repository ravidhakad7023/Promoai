"""LLM layer: Gemini REST provider (fake transport), cache, JSON validation + retry."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from promoai.config import ConfigError, Settings
from promoai.llm import (
    CachedProvider, FakeProvider, LLMBlockedError, LLMError, LLMQuotaError, LLMResponseError,
    create_provider, extract_json, generate_validated,
)
from promoai.llm_gemini import GeminiProvider

KEY = "AIza-TEST-SECRET-KEY"


def settings(**kw) -> Settings:
    base = dict(provider="gemini", model="test-model", api_key=KEY, min_interval_seconds=0.0, max_transport_retries=3)
    base.update(kw)
    return Settings(**base)


def ok_body(text="hello", **usage) -> bytes:
    u = {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "totalTokenCount": 17, **usage}
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}], "usageMetadata": u}).encode()


class Script:
    """Fake HTTP transport: returns queued (status, body) tuples and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, method, url, headers, body, timeout):
        self.requests.append((method, url, headers, json.loads(body) if body else None))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def gemini(transport, **kw):
    sleeps = []
    p = GeminiProvider(settings(**kw), transport=transport, sleep=sleeps.append)
    return p, sleeps


# ------------------------------------------------------------------ Gemini REST
def test_successful_call_parses_text_and_usage():
    t = Script((200, ok_body("the answer")))
    p, _ = gemini(t)
    assert p.generate("prompt") == "the answer"
    assert p.stats.api_calls == 1
    assert (p.stats.prompt_tokens, p.stats.output_tokens, p.stats.thinking_tokens, p.stats.total_tokens) == (10, 5, 2, 17)


def test_key_is_in_header_never_in_url_or_body():
    t = Script((200, ok_body()))
    p, _ = gemini(t)
    p.generate("secret screenplay words")
    method, url, headers, body = t.requests[0]
    assert method == "POST" and url.endswith("/models/test-model:generateContent")
    assert headers["x-goog-api-key"] == KEY
    assert KEY not in url and KEY not in json.dumps(body)


def test_no_provider_specific_structured_output_features():
    t = Script((200, ok_body()))
    p, _ = gemini(t)
    p.generate("x")
    cfg = t.requests[0][3]["generationConfig"]
    assert set(cfg) == {"temperature", "maxOutputTokens"}          # no responseSchema / responseMimeType
    assert "responseSchema" not in json.dumps(t.requests[0][3])


def test_thought_parts_are_excluded():
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "answer"}]}}]}).encode()
    p, _ = gemini(Script((200, body)))
    assert p.generate("x") == "answer"


def test_429_with_retry_delay_waits_then_succeeds():
    err = json.dumps({"error": {"code": 429, "message": "slow down", "details": [{"@type": "RetryInfo", "retryDelay": "7s"}]}}).encode()
    p, sleeps = gemini(Script((429, err), (200, ok_body("fine"))))
    assert p.generate("x") == "fine"
    assert sleeps == [8.0]                                          # 7s + 1s margin
    assert p.stats.api_calls == 1
    assert [f["kind"] for f in p.stats.failures] == ["http_429"]


def test_daily_quota_is_not_retried():
    err = json.dumps({"error": {"code": 429, "message": "quota", "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}).encode()
    t = Script((429, err))
    p, sleeps = gemini(t)
    with pytest.raises(LLMQuotaError):
        p.generate("x")
    assert len(t.requests) == 1 and sleeps == []
    assert p.stats.failures[0]["kind"] == "quota_daily"


def test_server_errors_retry_with_backoff_then_give_up():
    t = Script((503, b'{"error":{"message":"overloaded"}}'))
    p, sleeps = gemini(t)
    with pytest.raises(LLMError, match="persisted after 3 retries"):
        p.generate("x")
    assert len(t.requests) == 4 and sleeps == [2.0, 4.0, 8.0]


def test_network_failure_is_retried():
    p, sleeps = gemini(Script((0, b""), (200, ok_body("back"))))
    assert p.generate("x") == "back" and len(sleeps) == 1


def test_404_gives_model_hint_and_does_not_retry():
    t = Script((404, b'{"error":{"message":"model not found"}}'))
    p, _ = gemini(t)
    with pytest.raises(LLMError, match="--list-models"):
        p.generate("x")
    assert len(t.requests) == 1


def test_api_key_echoed_in_error_is_redacted():
    t = Script((400, json.dumps({"error": {"message": f"bad key {KEY}"}}).encode()))
    p, _ = gemini(t)
    with pytest.raises(LLMError) as e:
        p.generate("x")
    assert KEY not in str(e.value) and "<redacted>" in str(e.value)
    assert all(KEY not in json.dumps(f) for f in p.stats.failures)


def test_prompt_block_raises_blocked():
    body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()
    p, _ = gemini(Script((200, body)))
    with pytest.raises(LLMBlockedError):
        p.generate("x")
    assert p.stats.failures[-1]["kind"] == "blocked"


def test_safety_stopped_reply_with_no_text_raises_blocked():
    body = json.dumps({"candidates": [{"finishReason": "SAFETY"}]}).encode()
    p, _ = gemini(Script((200, body)))
    with pytest.raises(LLMBlockedError):
        p.generate("x")


def test_truncated_reply_is_returned_but_recorded():
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": '{"a": '}]}, "finishReason": "MAX_TOKENS"}]}).encode()
    p, _ = gemini(Script((200, body)))
    assert p.generate("x") == '{"a": '
    assert p.stats.failures[-1]["kind"] == "truncated"


def test_min_interval_throttles_calls():
    now = [100.0]
    sleeps = []
    p = GeminiProvider(settings(min_interval_seconds=6.0), transport=Script((200, ok_body())),
                       sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)), clock=lambda: now[0])
    p.generate("a")
    now[0] += 2.0
    p.generate("b")
    assert sleeps == [pytest.approx(4.0)]


def test_missing_key_or_model_is_a_config_error():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        GeminiProvider(settings(api_key=""))
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        GeminiProvider(settings(model=""))
    GeminiProvider(settings(model=""), require_model=False)


def test_list_models_filters_generate_content():
    body = json.dumps({"models": [
        {"name": "models/m-b", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/m-a", "supportedGenerationMethods": ["generateContent", "countTokens"]}]}).encode()
    p, _ = gemini(Script((200, body)))
    assert p.list_models() == ["m-a", "m-b"]


def test_factory_builds_cached_gemini_and_rejects_unknown(tmp_path):
    p = create_provider(settings(cache_dir=tmp_path))
    assert isinstance(p, CachedProvider) and p.name == "gemini" and p.model == "test-model"
    with pytest.raises(ConfigError, match="unknown LLM_PROVIDER"):
        create_provider(Settings(provider="nope"))


# ------------------------------------------------------------------ cache
def test_cache_hit_avoids_second_call(tmp_path):
    inner = FakeProvider(["answer"])
    c = CachedProvider(inner, tmp_path)
    assert c.generate("p") == "answer"
    assert c.generate("p") == "answer"
    assert len(inner.prompts) == 1
    assert c.stats.api_calls == 1 and c.stats.cache_hits == 1
    assert c.stats.tokens_saved_by_cache > 0


def test_cache_persists_across_instances(tmp_path):
    CachedProvider(FakeProvider(["answer"]), tmp_path).generate("p")
    inner2 = FakeProvider(["different"])
    assert CachedProvider(inner2, tmp_path).generate("p") == "answer"
    assert inner2.prompts == []


def test_cache_key_depends_on_prompt_and_model(tmp_path):
    a = CachedProvider(FakeProvider(["1", "2"], model="m1"), tmp_path)
    a.generate("p")
    assert a.generate("q") == "2"                                    # different prompt: miss
    b = CachedProvider(FakeProvider(["other"], model="m2"), tmp_path)
    assert b.generate("p") == "other"                                # different model: miss


def test_cache_stores_responses_not_prompts(tmp_path):
    CachedProvider(FakeProvider(["resp"]), tmp_path).generate("UNIQUE-PROMPT-MARKER confidential")
    blob = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "UNIQUE-PROMPT-MARKER" not in blob and "resp" in blob


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    inner = FakeProvider(["fresh"])
    c = CachedProvider(inner, tmp_path)
    c.generate("p")
    for f in tmp_path.rglob("*.json"):
        f.write_text("{not json")
    assert CachedProvider(FakeProvider(["again"]), tmp_path).generate("p") == "again"


def test_invalidate_and_remember(tmp_path):
    inner = FakeProvider(["first", "second"])
    c = CachedProvider(inner, tmp_path)
    c.generate("p")
    c.invalidate("p")
    assert c.generate("p") == "second"
    c.remember("p", "pinned")
    assert c.generate("p") == "pinned"


def test_cache_delegates_provider_methods(tmp_path):
    p, _ = gemini(Script((200, json.dumps({"models": []}).encode())))
    assert CachedProvider(p, tmp_path).list_models() == []


# ------------------------------------------------------------------ JSON extraction + validation
class Item(BaseModel):
    name: str
    n: int


@pytest.mark.parametrize("text", [
    '{"name": "a", "n": 1}',
    '```json\n{"name": "a", "n": 1}\n```',
    '```\n{"name": "a", "n": 1}\n```',
    'Sure! Here you go:\n{"name": "a", "n": 1}\nHope that helps.',
])
def test_extract_json_tolerates_fences_and_chatter(text):
    assert extract_json(text) == {"name": "a", "n": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_reply_needs_one_call():
    p = FakeProvider(['{"name": "a", "n": 1}'])
    obj, attempts = generate_validated(p, "prompt", Item)
    assert obj == Item(name="a", n=1) and attempts == 1 and len(p.prompts) == 1
    assert p.stats.json_retries == 0


def test_invalid_json_is_retried_once_with_error_feedback():
    p = FakeProvider(['{"name": "a", "n": "oops"}', '{"name": "a", "n": 2}'])
    obj, attempts = generate_validated(p, "ORIGINAL", Item)
    assert obj.n == 2 and attempts == 2 and len(p.prompts) == 2
    assert p.prompts[1].startswith("ORIGINAL")
    assert "n:" in p.prompts[1] and '"n": "oops"' in p.prompts[1]   # error path + previous reply shown
    assert p.stats.json_retries == 1


def test_unparseable_reply_is_retried():
    p = FakeProvider(["total nonsense", '{"name": "a", "n": 3}'])
    assert generate_validated(p, "x", Item)[0].n == 3


def test_second_invalid_reply_raises_and_does_not_retry_again():
    p = FakeProvider(["bad", '{"name": 1}'])
    with pytest.raises(LLMResponseError):
        generate_validated(p, "x", Item)
    assert len(p.prompts) == 2                                      # exactly one retry
    assert [f["kind"] for f in p.stats.failures] == ["invalid_json", "invalid_json_after_retry"]


def test_transport_errors_propagate_unchanged():
    def boom(_):
        raise LLMQuotaError("daily")
    with pytest.raises(LLMQuotaError):
        generate_validated(FakeProvider(boom), "x", Item)


def test_bad_first_reply_is_not_left_in_cache_and_repaired_reply_is_remembered(tmp_path):
    inner = FakeProvider(["garbage", '{"name": "fixed", "n": 1}'])
    c = CachedProvider(inner, tmp_path)
    obj, attempts = generate_validated(c, "P", Item)
    assert obj.name == "fixed" and attempts == 2 and len(inner.prompts) == 2
    # next run, brand-new provider: the ORIGINAL prompt is a cache hit -> zero API calls
    inner2 = FakeProvider(["should not be used"])
    obj2, attempts2 = generate_validated(CachedProvider(inner2, tmp_path), "P", Item)
    assert obj2.name == "fixed" and attempts2 == 1 and inner2.prompts == []


def test_failed_validation_leaves_no_cache_entry(tmp_path):
    c = CachedProvider(FakeProvider(["bad", "still bad"]), tmp_path)
    with pytest.raises(LLMResponseError):
        generate_validated(c, "P", Item)
    assert list(tmp_path.rglob("*.json")) == []
