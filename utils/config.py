import os
import tomllib

# Every pipeline setting: name -> (type, default). Names match the environment variables;
# in config.toml the same names are written in lower case.
# Secrets (OPENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, HF_TOKEN) are read from the environment / .env only.
SETTINGS = {
    # languages
    "SOURCE_LANGUAGE": (str, ""),          # empty = detect
    "TARGET_LANGUAGE": (str, "en"),
    # separation
    "DEMUCS_MODEL": (str, "htdemucs_ft"),
    "DEMUCS_DEVICE": (str, ""),
    "DEMUCS_SHIFTS": (int, 1),
    "SEED": (int, 0),                      # makes separation repeatable
    # transcription
    "ASR_BACKEND": (str, "whisperx"),
    "WHISPER_MODEL": (str, "large-v3-turbo"),
    "ASR_DEVICE": (str, ""),
    # diarization
    "NUM_SPEAKERS": (int, None),
    "MIN_SPEAKERS": (int, None),
    "MAX_SPEAKERS": (int, None),
    "DIARIZATION_DEVICE": (str, ""),
    # translation
    "TRANSLATOR": (str, "google"),
    "GLOSSARY_FILE": (str, ""),
    "VIDEO_CONTEXT": (str, ""),
    "LLM_MODEL": (str, "gpt-5-mini"),
    "LLM_BASE_URL": (str, ""),
    "NLLB_MODEL": (str, "facebook/nllb-200-distilled-600M"),
    "TRANSLATION_DEVICE": (str, ""),
    # voice cloning
    "TTS_BACKEND": (str, "f5"),
    "F5_MODEL": (str, "F5TTS_v1_Base"),
    "TTS_DEVICE": (str, ""),
    "TTS_VERIFY": (bool, True),
    "TTS_ATTEMPTS": (int, 3),
    "TTS_ON_MISMATCH": (str, "auto"),      # auto | original | silence | keep
    # timing fit
    "FIT_MIN_SPEED": (float, 0.9),
    "FIT_MAX_SPEED": (float, 1.25),
    "FIT_HARD_MAX_SPEED": (float, 1.6),
    "FIT_MAX_BORROW": (float, 0.5),
    "FIT_MAX_RETRANSLATIONS": (int, 2),
    # mixing
    "MIX_TARGET_LUFS": (float, -16.0),
    "MIX_TRUE_PEAK": (float, -1.5),
    "MIX_DUCK_DB": (float, -6.0),
    "MIX_KEEP_VOCALS_DB": (float, None),   # empty = off
    # final video
    "MUX_KEEP_ORIGINAL_AUDIO": (bool, True),
    "MUX_ORIGINAL_SUBTITLES": (bool, True),
    "MUX_AUDIO_BITRATE": (str, "192k"),
}

# Settings that translator / TTS backends read from the environment themselves
BACKEND_SETTINGS = ("LLM_MODEL", "LLM_BASE_URL", "NLLB_MODEL", "TRANSLATION_DEVICE", "F5_MODEL", "TTS_DEVICE")


def load_config(config_path: str | None = None, overrides: dict | None = None) -> dict:
    """
    Resolve every setting. Later sources win: defaults < config file < environment (.env) < overrides
    (command-line flags). config_path defaults to ./config.toml when it exists.
    """
    config = {name: default for name, (_, default) in SETTINGS.items()}

    path = config_path or ("config.toml" if os.path.exists("config.toml") else None)
    if path:
        with open(path, "rb") as f:
            file_values = tomllib.load(f)
        unknown = [k for k in file_values if k.upper() not in SETTINGS]
        if unknown:
            raise ValueError(f"Unknown settings in {path}: {', '.join(unknown)}")
        for key, value in file_values.items():
            config[key.upper()] = _convert(key.upper(), value)

    for name in SETTINGS:
        if os.getenv(name) not in (None, ""):
            config[name] = _convert(name, os.environ[name])

    for name, value in (overrides or {}).items():
        if value is not None:
            config[name] = _convert(name, value)

    config["TRANSLATOR"] = config["TRANSLATOR"].lower()
    config["TTS_BACKEND"] = config["TTS_BACKEND"].lower()
    config["TTS_ON_MISMATCH"] = config["TTS_ON_MISMATCH"].lower()
    for name in BACKEND_SETTINGS:
        if config[name]:
            os.environ[name] = str(config[name])
        else:
            os.environ.pop(name, None)
    return config


def _convert(name: str, value):
    kind, default = SETTINGS[name]
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    if kind is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() not in ("false", "0", "no", "off")
    return kind(value)
