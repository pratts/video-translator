from utils.segment import build_segments


def words(*items):
    return [{"word": w, "start": s, "end": e, "score": 0.9} for w, s, e in items]


def transcript(blocks, language="en"):
    return {"language": language,
            "segments": [{"id": i, "start": b[0]["start"], "end": b[-1]["end"], "text": "", "words": b}
                         for i, b in enumerate(blocks)]}


def test_splits_at_speaker_change_and_sentence_end():
    t = transcript([words(("Hello", 0.0, 0.4), ("there.", 0.45, 0.9), ("How", 1.0, 1.3), ("are", 1.35, 1.6),
                          ("you?", 1.65, 2.0), ("Fine", 2.5, 2.9), ("thanks.", 2.95, 3.5))])
    turns = [{"start": 0.0, "end": 2.1, "speaker": "SPEAKER_00"}, {"start": 2.4, "end": 3.6, "speaker": "SPEAKER_01"}]
    doc = build_segments(t, turns, video_duration=5.0, min_duration=0.5)
    assert [(s["speaker"], s["text"]) for s in doc["segments"]] == [
        ("SPEAKER_00", "Hello there."), ("SPEAKER_00", "How are you?"), ("SPEAKER_01", "Fine thanks.")]
    assert doc["segments"][-1]["gap_after"] == 1.5


def test_short_speaker_flicker_is_absorbed():
    t = transcript([words(("one", 0.0, 0.5), ("two", 0.5, 1.0), ("three", 1.0, 1.1), ("four", 1.1, 1.6))])
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}, {"start": 1.0, "end": 1.1, "speaker": "B"},
             {"start": 1.1, "end": 1.6, "speaker": "A"}]
    doc = build_segments(t, turns)
    assert doc["speakers"] == ["A"]


def test_japanese_punctuation_attaches_and_long_segments_split():
    chars = [(c, i * 0.25, i * 0.25 + 0.25) for i, c in enumerate("今日はいい天気ですね")]
    chars.insert(3, ("、", 0.75, 0.75))
    t = transcript([words(*chars)], language="ja")
    doc = build_segments(t, [{"start": 0, "end": 5, "speaker": "S"}], max_duration=1.5, min_duration=0.5)
    assert all(not s["text"].startswith("、") for s in doc["segments"])
    assert all(s["duration"] <= 1.5 for s in doc["segments"])
    assert "".join(s["text"] for s in doc["segments"]) == "今日は、いい天気ですね"
