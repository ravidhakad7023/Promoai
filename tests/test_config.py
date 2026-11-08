import pytest

from promoai.config import ConfigError, Settings, load_dotenv, load_settings


def test_dotenv_parsing(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# c\nGEMINI_API_KEY="abc 123"\nLLM_MODEL=m1  # note\nexport LLM_PROVIDER=gemini\n\nbad line\nEMPTY=\n', encoding="utf-8")
    assert load_dotenv(f) == {"GEMINI_API_KEY": "abc 123", "LLM_MODEL": "m1", "LLM_PROVIDER": "gemini", "EMPTY": ""}


def test_missing_dotenv_is_fine(tmp_path):
    assert load_dotenv(tmp_path / "nope") == {}


def test_environment_overrides_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text("LLM_MODEL=from-file\nGEMINI_API_KEY=file-key\n")
    s = load_settings(f, environ={"LLM_MODEL": "from-env"})
    assert s.model == "from-env" and s.api_key == "file-key"


def test_defaults(tmp_path):
    s = load_settings(tmp_path / "none", environ={})
    assert (s.provider, s.model, s.api_key) == ("gemini", "", "")
    assert s.temperature == 0.4 and s.min_interval_seconds == 6.0


def test_provider_selects_key_variable(tmp_path):
    s = load_settings(tmp_path / "x", environ={"LLM_PROVIDER": "Other", "OTHER_API_KEY": "k", "GEMINI_API_KEY": "g"})
    assert s.provider == "other" and s.api_key == "k"


def test_bad_number_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="LLM_TEMPERATURE"):
        load_settings(tmp_path / "x", environ={"LLM_TEMPERATURE": "hot"})


def test_secret_never_appears_in_repr():
    s = Settings(api_key="SUPER-SECRET-KEY", model="m")
    assert "SUPER-SECRET-KEY" not in repr(s) and "SUPER-SECRET-KEY" not in str(s)
