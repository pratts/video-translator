import json

import pytest
import soundfile as sf

from utils.fit import _plan, fit_clips


def test_plan():
    assert _plan(0.9, 1.0, 1.5, 0.9, 1.25) == pytest.approx((0.9, 1.0))   # slow down to fill the slot
    assert _plan(0.5, 1.0, 1.5, 0.9, 1.25)[0] == pytest.approx(0.9)       # but never below min_speed
    assert _plan(1.2, 1.0, 1.5, 0.9, 1.25) == (1.2, 1.0)             # speed up within the slot
    assert _plan(1.4, 1.0, 1.5, 0.9, 1.25) == (1.0, 1.4)             # use the following silence as is
    speed, target = _plan(2.0, 1.0, 1.5, 0.9, 1.25)                  # speed up into slot + silence
    assert target == 1.5 and round(speed, 3) == 1.333


def clip(i, start, duration, audio, text="hello"):
    return {"id": i, "speaker": "A", "start": start, "duration": duration, "clip_duration": 0, "text": text,
            "audio": audio}


def test_clips_never_overlap_and_silence_is_trimmed(tone, tmp_path):
    long_clip = tone(3.0, "long.wav", pad=0.4)
    short_clip = tone(0.5, "short.wav", pad=0.3)
    clips = {"clips": [clip(0, 0.0, 1.0, long_clip), clip(1, 1.3, 1.0, short_clip), clip(2, 5.0, 1.0, long_clip)]}
    doc = fit_clips(clips, str(tmp_path / "fitted"), video_duration=6.0)
    report = doc["clips"]
    assert doc["overlaps"] == []
    assert report[0]["end"] <= 1.3 - 0.1 + 0.001          # stops min_gap before the next clip
    assert report[0]["warning"]                           # needed more than hard_max_speed
    assert report[1]["trimmed"] < 0.75                    # most of the 2 x 0.3 s padding was trimmed (1.1 s before)
    assert report[2]["end"] <= 6.0
    for r in report:
        audio, sr = sf.read(r["audio"])
        assert abs(len(audio) / sr - r["fitted"]) < 0.002
    saved = json.load(open(tmp_path / "fitted" / "timing_report.json"))
    assert saved["clips"][1]["id"] == 1


def test_too_long_clip_is_shortened_by_regenerate(tone, tmp_path):
    calls = []
    short = tone(1.0, "regen.wav")

    def regenerate(c, max_chars, window):
        calls.append((max_chars, window))
        audio, sr = sf.read(short, dtype="float32")
        return audio, sr, "short"

    clips = {"clips": [clip(0, 0.0, 1.0, tone(3.0, "long.wav"), text="a very long line indeed")]}
    doc = fit_clips(clips, str(tmp_path / "fitted"), video_duration=10.0, regenerate=regenerate)
    assert len(calls) == 1 and calls[0][1] == 1.5
    assert calls[0][0] < len("a very long line indeed")
    assert doc["clips"][0]["retranslations"] == 1 and doc["clips"][0]["text"] == "short"
    assert doc["clips"][0]["warning"] is None
