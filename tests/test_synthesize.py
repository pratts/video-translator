import json
import types

import numpy as np
import pytest

import utils.synthesize as syn


def test_word_error_rate():
    assert syn.word_error_rate("Hello, world!", "hello world") == 0.0
    assert syn.word_error_rate("EasyJapanese is here", "Easy Japanese is here.") == 0.0
    assert syn.word_error_rate("I'm Mana Kumagai.", "I'm Manakumagai.") == 0.0
    assert syn.word_error_rate("Japan's best parts", "Japan's best part?") == 1 / 3
    assert syn.word_error_rate("one two", "") == 1.0


def test_retries_until_heard_and_falls_back_on_failure(monkeypatch, tmp_path):
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
    assert second["status"] == "fallback_original" and second["clip_duration"] == 0.5   # short line: auto
    out = syn.synthesize_segments(doc, {"A": {}}, str(tmp_path), attempts=3, on_mismatch="keep")
    assert out["clips"][1]["status"] == "failed"                  # nothing to keep: silence


def test_missed_words_catches_wrong_content_words_only():
    assert syn.missed_words("What do you think are Japan's strengths?",
                            "What do you think are Japan's drinks?") == ["strengths"]
    assert syn.missed_words("Hi. Today, uh, about Japan", "Hi, today, ah, about Japan.") == []   # short filler
    assert syn.missed_words("I'm Mana Kumagai from Sendai", "I'm Manakumagai from Sendai") == []
    assert syn.missed_words("I'm Mana Kumagai", "I'm Mana Kumagae") == []                         # near spelling
    assert syn.missed_words("the freshest fish", "the fish") == ["freshest"]
    assert syn.extra_words("For fish, Japan has the freshest fish.", "Sakana, for fish Japan has the freshest fish.") \
        == ["+sakana"]
    assert syn.extra_words("Easy Japanese is coming to you from Sendai.", "EasyJapanese is coming from Sendai, uh") == []


def stub_backend(monkeypatch, heard):
    """A TTS backend whose clips are always transcribed as heard(text)."""
    backend = types.SimpleNamespace(SAMPLE_RATE=24000, SUPPORTS_DURATION=True, texts=[])

    def synthesize(text, reference, language, target_duration=None, seed=0):
        backend.texts.append(text)
        return np.full(int(24000 * target_duration), len(backend.texts), np.float32), 24000

    backend.synthesize = synthesize
    monkeypatch.setattr(syn, "get_synthesizer", lambda name: backend)
    monkeypatch.setattr(syn, "_transcribe", lambda wav, sr, lang, m: heard(backend.texts[int(wav[0]) - 1]))
    return backend


def translation_doc(*durations):
    return {"translator": "stub", "source_language": "ja", "target_language": "en", "glossary": {}, "context": "",
            "segments": [{"id": i, "speaker": "A", "start": 2.0 * i, "end": 2.0 * i + d, "duration": d,
                          "gap_after": 0.5, "source_text": "日本の強み", "raw": "Japan's strengths",
                          "translation": "Japan's strengths", "max_chars": 30, "fits": True}
                         for i, d in enumerate(durations)]}


def stub_translator(monkeypatch, rephrased):
    import utils.translate as tr
    calls = []

    def rephrase(item, translation, heard, max_chars, *args, **kwargs):
        calls.append((translation, heard))
        return rephrased

    monkeypatch.setattr(tr, "get_translator", lambda name: types.SimpleNamespace(__name__="stub", rephrase=rephrase))
    return calls


def test_line_never_heard_is_rephrased_once_then_falls_back(monkeypatch, tmp_path):
    backend = stub_backend(monkeypatch, lambda text: "Japan's drinks")
    calls = stub_translator(monkeypatch, "What Japan is good at")
    doc = translation_doc(0.8, 2.0)
    path = tmp_path / "translation" / "translation.json"
    out = syn.synthesize_segments(doc, {"A": {}}, str(tmp_path / "clips"), attempts=2, translation_path=str(path))
    short, long = out["clips"]
    assert len(calls) == 2 and calls[0] == ("Japan's strengths", "Japan's drinks")
    assert backend.texts.count("What Japan is good at") == 4              # 2 seeds per line after rephrasing
    assert short["status"] == "fallback_original" and long["status"] == "fallback_silence"   # auto policy
    assert short["attempts"] == 4 and short["rephrased_from"] == "Japan's strengths"
    assert short["text"] == "What Japan is good at" and short["missed"]
    assert json.loads(path.read_text())["segments"][0]["translation"] == "What Japan is good at"
    assert [line.split(":")[0] for line in syn.fallback_summary(out)] == ["segment 0 (0.00s, A)", "segment 1 (2.00s, A)"]


def test_rephrased_line_that_is_heard_is_used(monkeypatch, tmp_path):
    stub_backend(monkeypatch, lambda text: "Japan's drinks" if "strengths" in text else text)
    stub_translator(monkeypatch, "What Japan is good at")
    out = syn.synthesize_segments(translation_doc(2.0), {"A": {}}, str(tmp_path / "clips"), attempts=3,
                                  translation_path=str(tmp_path / "translation.json"))
    clip = out["clips"][0]
    assert clip["status"] == "ok" and clip["attempts"] == 4 and clip["text"] == "What Japan is good at"
    assert "rephrased" in syn.fallback_summary(out)[0]


@pytest.mark.parametrize("policy, status", [("keep", "mismatch"), ("original", "fallback_original"),
                                            ("silence", "fallback_silence")])
def test_fallback_policy_without_rephrasing(monkeypatch, tmp_path, policy, status):
    backend = stub_backend(monkeypatch, lambda text: "Japan's drinks")
    out = syn.synthesize_segments(translation_doc(0.8), {"A": {}}, str(tmp_path), attempts=2, on_mismatch=policy)
    assert out["clips"][0]["status"] == status and len(backend.texts) == 2   # no translation_path: no rephrase
    with pytest.raises(ValueError):
        syn.synthesize_segments(translation_doc(0.8), {"A": {}}, str(tmp_path), on_mismatch="maybe")


def test_shortened_line_that_is_not_heard_is_discarded(monkeypatch, tmp_path):
    import utils.translate as tr
    stub_backend(monkeypatch, lambda text: "something else")
    monkeypatch.setattr(tr, "get_translator", lambda name: types.SimpleNamespace(
        __name__="stub", shorten=lambda item, translation, max_chars, *a, **k: "Japan's best"))
    doc = translation_doc(2.0)
    path = tmp_path / "translation.json"
    regenerate = syn.make_regenerator(doc, str(path), {"A": {}}, attempts=2)
    assert regenerate({"id": 0, "speaker": "A"}, 12, 2.5) is None
    assert doc["segments"][0]["translation"] == "Japan's strengths" and not path.exists()


def padding_backend(monkeypatch, overhead=0.8):
    """Like F5: a clip of exactly target_duration, of which overhead seconds are silence at the edges."""
    backend = types.SimpleNamespace(SAMPLE_RATE=24000, SUPPORTS_DURATION=True, requests=[])

    def synthesize(text, reference, language, target_duration=None, seed=0):
        backend.requests.append((round(target_duration, 3), seed))
        sr, pad = 24000, int(overhead / 2 * 24000)
        speech = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int((target_duration - overhead) * sr)) / sr)
        return np.concatenate([np.zeros(pad), speech, np.zeros(pad)]).astype(np.float32), sr

    backend.synthesize = synthesize
    monkeypatch.setattr(syn, "get_synthesizer", lambda name: backend)
    monkeypatch.setattr(syn, "_transcribe", lambda wav, sr, lang, m: "Japan's strengths")
    return backend


def test_short_speech_is_generated_again_with_more_room(monkeypatch, tmp_path):
    backend = padding_backend(monkeypatch)
    clip = syn.synthesize_segments(translation_doc(3.0), {"A": {}}, str(tmp_path))["clips"][0]
    # speech ~2.3 s of 3 s: half the missing time first (still under 90%), then three quarters, another seed
    (first, _), (half, seed1), (three_quarters, seed2) = backend.requests
    assert first == 3.0 < half < three_quarters < 3.8 and (seed1, seed2) == (0, 1)
    assert clip["lengthened"] and 0.9 <= clip["coverage"] <= 1.25
    assert clip["attempts"] == 3 and clip["status"] == "ok"


def test_lengthening_skips_short_lines_and_never_exceeds_max_speed(monkeypatch, tmp_path):
    backend = padding_backend(monkeypatch, overhead=0.4)
    clip = syn.synthesize_segments(translation_doc(0.9), {"A": {}}, str(tmp_path))["clips"][0]
    assert len(backend.requests) == 1 and not clip["lengthened"]          # under 1 s: left as is

    # a backend whose speech always overshoots: longer takes are generated (one per seed) but not used
    backend = padding_backend(monkeypatch, overhead=1.0)
    real = backend.synthesize
    backend.synthesize = lambda text, ref, lang, target_duration=None, seed=0: real(
        text, ref, lang, target_duration * (1 if target_duration == 3.0 else 1.8), seed)
    clip = syn.synthesize_segments(translation_doc(3.0), {"A": {}}, str(tmp_path), max_speed=1.25)["clips"][0]
    assert len(backend.requests) == 4 and not clip["lengthened"] and clip["coverage"] < 0.9


def test_lengthened_take_must_still_be_heard(monkeypatch, tmp_path):
    backend = padding_backend(monkeypatch)
    # longer takes are misheard with seed 0 and heard with seed 1
    monkeypatch.setattr(syn, "_transcribe", lambda wav, sr, lang, m: "Japan's strengths"
                        if len(wav) <= 3 * sr or backend.requests[-1][1] == 1 else "Japan's drinks")
    clip = syn.synthesize_segments(translation_doc(3.0), {"A": {}}, str(tmp_path))["clips"][0]
    assert [seed for _, seed in backend.requests] == [0, 0, 1]
    assert clip["lengthened"] and clip["attempts"] == 3 and clip["coverage"] >= 0.9

    backend.requests.clear()
    monkeypatch.setattr(syn, "_transcribe", lambda wav, sr, lang, m: "Japan's strengths" if len(wav) <= 3 * sr
                        else "Japan's drinks")
    clip = syn.synthesize_segments(translation_doc(3.0), {"A": {}}, str(tmp_path))["clips"][0]
    assert len(backend.requests) == 4 and not clip["lengthened"] and clip["heard"] == "Japan's strengths"
