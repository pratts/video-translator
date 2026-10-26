from utils.config import load_config


def test_precedence_defaults_file_env_flags(tmp_path, monkeypatch):
    for name in ("TRANSLATOR", "TARGET_LANGUAGE", "TTS_VERIFY", "NUM_SPEAKERS", "FIT_MAX_SPEED", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('translator = "nllb"\ntarget_language = "es"\ntts_verify = false\nnum_speakers = 2\n'
                           'llm_model = "qwen2.5:14b"\n')
    monkeypatch.setenv("TARGET_LANGUAGE", "fr")
    monkeypatch.setenv("FIT_MAX_SPEED", "1.4")
    config = load_config(str(config_file), {"TRANSLATOR": "GPT", "TTS_BACKEND": None})
    assert config["TRANSLATOR"] == "gpt"          # flag beats file
    assert config["TARGET_LANGUAGE"] == "fr"      # environment beats file
    assert config["TTS_VERIFY"] is False and config["NUM_SPEAKERS"] == 2
    assert config["FIT_MAX_SPEED"] == 1.4
    assert config["TTS_BACKEND"] == "f5"          # default
    assert config["MIX_KEEP_VOCALS_DB"] is None
    import os
    assert os.environ["LLM_MODEL"] == "qwen2.5:14b"   # exported for the backend that reads it


def test_unknown_setting_is_an_error(tmp_path):
    import pytest
    bad = tmp_path / "config.toml"
    bad.write_text('translater = "gpt"\n')
    with pytest.raises(ValueError, match="translater"):
        load_config(str(bad))
