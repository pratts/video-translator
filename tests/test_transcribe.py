import types

import numpy as np

import utils.transcribe as tr


def test_asr_prompt_lists_the_glossary_terms():
    assert tr.asr_prompt(None) is None and tr.asr_prompt({}) is None
    assert tr.asr_prompt({"熊谷真奈": "Mana Kumagai", "EasyJapanese": "Easy Japanese", "仙台": "Sendai"}) == \
        "熊谷真奈、EasyJapanese、仙台。"
    assert tr.asr_prompt({"Kumagai": "熊谷", " Sendai ": "仙台"}) == "Kumagai, Sendai."


def test_prompt_reaches_whisperx(monkeypatch, tmp_path):
    seen = {}

    def load_model(name, device, compute_type, language, asr_options):
        seen.update(asr_options)
        return types.SimpleNamespace(transcribe=lambda audio, batch_size, language: {
            "language": "ja", "segments": [{"start": 0.0, "end": 1.0, "text": "熊谷真奈です"}]})

    monkeypatch.setattr(tr.whisperx, "load_audio", lambda path, sr: np.zeros(16000, np.float32))
    monkeypatch.setattr(tr.whisperx, "load_model", load_model)
    monkeypatch.setattr(tr.whisperx, "load_align_model", lambda language_code, device: (None, None))
    monkeypatch.setattr(tr.whisperx, "align", lambda segments, *a, **k: {"segments": [
        {**s, "words": [{"word": "熊", "start": 0.0, "end": 0.2, "score": 0.9}]} for s in segments]})
    tr.transcribe("v.wav", str(tmp_path / "t.json"), str(tmp_path / "t.srt"), device="cpu",
                  glossary={"熊谷真奈": "Mana Kumagai"})
    assert seen["initial_prompt"] == "熊谷真奈。"
