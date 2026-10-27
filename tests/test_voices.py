import numpy as np
import soundfile as sf

import utils.voices as voices


def audio_files(tmp_path, seconds=30.0, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(tmp_path / "vocals.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    sf.write(tmp_path / "background.wav", (0.01 * np.sin(2 * np.pi * 60 * t)).astype(np.float32), sr)
    return str(tmp_path / "vocals.wav"), str(tmp_path / "background.wav")


def segment(i, speaker, start, end, text):
    return {"id": i, "speaker": speaker, "start": start, "end": end, "duration": end - start, "text": text,
            "words": [{"word": text, "start": start, "end": end, "score": 0.9}]}


def segments_doc(*segments):
    return {"language": "ja", "speakers": sorted({s["speaker"] for s in segments}), "segments": list(segments)}


def test_short_lines_are_used_only_when_a_speaker_has_too_little_speech(tmp_path, capsys):
    doc = segments_doc(segment(0, "HOST", 0.0, 4.0, "一。"), segment(1, "HOST", 5.0, 9.0, "二。"),
                       segment(2, "HOST", 10.0, 10.8, "はい。"),
                       segment(3, "GUEST", 12.0, 15.5, "魚。"), segment(4, "GUEST", 16.0, 16.8, "はい。"))
    refs = voices.build_voice_references(doc, [], *audio_files(tmp_path), str(tmp_path / "voices"))
    assert refs["HOST"]["segments"] == [0, 1]                 # 8 s without the short line
    assert sorted(refs["GUEST"]["segments"]) == [3, 4]        # 3.5 s is too little: the 0.8 s line is added
    assert refs["GUEST"]["source"] == "auto"
    out = capsys.readouterr().out
    assert "Only" in out and "GUEST" in out and "voice_references_dir" in out and "4.3s in all 2" in out


def test_manual_reference_replaces_the_automatic_one(tmp_path, monkeypatch):
    manual = tmp_path / "manual"
    manual.mkdir()
    sf.write(manual / "GUEST.wav", np.full(16000 * 6, 0.1, np.float32), 16000)
    (manual / "GUEST.txt").write_text("魚だったらやっぱりね。\n", encoding="utf-8")
    doc = segments_doc(segment(0, "HOST", 0.0, 8.0, "一。"), segment(1, "GUEST", 12.0, 15.5, "魚。"))
    monkeypatch.setattr(voices, "_transcribe", lambda audio, model: ("not used", "en"))
    refs = voices.build_voice_references(doc, [], *audio_files(tmp_path), str(tmp_path / "voices"),
                                         manual_dir=str(manual))
    guest = refs["GUEST"]
    assert guest["source"] == "manual" and guest["text"] == "魚だったらやっぱりね。" and guest["language"] == "ja"
    assert abs(guest["duration"] - 6.0) < 0.01
    assert sf.read(guest["audio"])[1] == voices.REFERENCE_SAMPLE_RATE
    assert refs["HOST"]["source"] == "auto"

    (manual / "GUEST.txt").unlink()                          # no transcript: Whisper writes one
    refs = voices.build_voice_references(doc, [], *audio_files(tmp_path), str(tmp_path / "voices"),
                                         manual_dir=str(manual))
    assert (refs["GUEST"]["text"], refs["GUEST"]["language"]) == ("not used", "en")
