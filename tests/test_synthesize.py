import types

import numpy as np

import utils.synthesize as syn


def test_word_error_rate():
    assert syn.word_error_rate("Hello, world!", "hello world") == 0.0
    assert syn.word_error_rate("EasyJapanese is here", "Easy Japanese is here.") == 0.0
    assert syn.word_error_rate("I'm Mana Kumagai.", "I'm Manakumagai.") == 0.0
    assert syn.word_error_rate("Japan's best parts", "Japan's best part?") == 1 / 3
    assert syn.word_error_rate("one two", "") == 1.0


def test_retries_until_heard_and_writes_silence_on_failure(monkeypatch, tmp_path):
    backend = types.SimpleNamespace(SAMPLE_RATE=24000, SUPPORTS_DURATION=True)
    seeds = []

    def synthesize(text, reference, language, target_duration=None, seed=0):
        if text == "boom":
            raise RuntimeError("model crashed")
        seeds.append(seed)
        return np.full(int(24000 * target_duration), seed, np.float32), 24000

    backend.synthesize = synthesize
    monkeypatch.setattr(syn, "get_synthesizer", lambda name: backend)
    monkeypatch.setattr(syn, "_transcribe", lambda wav, sr, lang, m: "hello there" if wav[0] == 1 else "nope")
    doc = {"target_language": "en", "segments": [
        {"id": 0, "speaker": "A", "start": 0, "duration": 1.0, "gap_after": 0.5, "translation": "hello there"},
        {"id": 1, "speaker": "A", "start": 2, "duration": 0.5, "gap_after": None, "translation": "boom"}]}
    out = syn.synthesize_segments(doc, {"A": {}}, str(tmp_path), attempts=3)
    first, second = out["clips"]
    assert seeds == [0, 1] and first["status"] == "ok" and first["attempts"] == 2
    assert second["status"] == "failed" and second["clip_duration"] == 0.5
