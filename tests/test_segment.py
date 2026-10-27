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


def chars(text, start=0.0, step=0.2):
    return [(c, round(start + i * step, 3), round(start + (i + 1) * step, 3)) for i, c in enumerate(text)]


def test_japanese_speaker_change_never_splits_a_word():
    # the diarization boundary (2.1 s) falls inside けど (2.0-2.4); the alignment is per character
    t = transcript([words(*chars("お聞きしてるんですけど日本のいいところ"))], language="ja")
    turns = [{"start": 0.0, "end": 2.1, "speaker": "HOST"}, {"start": 2.1, "end": 4.0, "speaker": "GUEST"}]
    doc = build_segments(t, turns, min_duration=0.5)
    assert [(s["speaker"], s["text"]) for s in doc["segments"]] == [
        ("HOST", "お聞きしてるんですけど"), ("GUEST", "日本のいいところ")]


def test_japanese_boundary_inside_a_noun_keeps_the_noun_together():
    t = transcript([words(*chars("お聞きしてるんですけど日本のいいところ"))], language="ja")
    turns = [{"start": 0.0, "end": 2.3, "speaker": "HOST"}, {"start": 2.3, "end": 4.0, "speaker": "GUEST"}]
    texts = [s["text"] for s in build_segments(t, turns, min_duration=0.5)["segments"]]
    assert all(not s.startswith(("ど", "本")) for s in texts)
    assert "".join(texts) == "お聞きしてるんですけど日本のいいところ"


def test_weakly_supported_fragment_is_absorbed():
    # "two" (0.7 s) is given to B by only 30% of its time; the rest of it is not covered at all
    t = transcript([words(("one", 0.0, 1.0), ("two", 1.0, 1.7), ("three", 1.7, 3.0))])
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}, {"start": 1.0, "end": 1.21, "speaker": "B"},
             {"start": 1.7, "end": 3.0, "speaker": "A"}]
    assert build_segments(t, turns)["speakers"] == ["A"]


def test_well_supported_short_turn_is_kept():
    t = transcript([words(("Really?", 0.0, 1.5), ("Yes.", 1.6, 2.4), ("Great.", 2.5, 4.0))])
    turns = [{"start": 0.0, "end": 1.5, "speaker": "A"}, {"start": 1.6, "end": 2.4, "speaker": "B"},
             {"start": 2.5, "end": 4.0, "speaker": "A"}]
    doc = build_segments(t, turns, min_duration=0.5)
    assert [s["speaker"] for s in doc["segments"]] == ["A", "B", "A"]


def test_chinese_speaker_change_never_splits_a_word():
    # the diarization boundary (0.9 s) falls inside 日本 (0.8-1.2)
    t = transcript([words(*chars("你觉得日本好在哪里"))], language="zh")
    turns = [{"start": 0.0, "end": 0.9, "speaker": "A"}, {"start": 0.9, "end": 2.0, "speaker": "B"}]
    texts = [s["text"] for s in build_segments(t, turns, min_duration=0.3, min_turn=0.1)["segments"]]
    assert not any(t.startswith("本") or t.endswith("日") for t in texts)
