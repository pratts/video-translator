from groupers import group_into_phrases
from groupers import chinese, japanese


def units(text, start=0.0, step=0.2, block=0):
    return [{"word": c, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3),
             "score": 0.9, "block": block} for i, c in enumerate(text)]


def test_japanese_phrases_keep_function_words_with_their_word():
    phrases = [p["word"] for p in group_into_phrases(units("お聞きしてるんですけど日本のいいところ"), "ja")]
    assert phrases == ["お聞きしてるんですけど", "日本の", "いい", "ところ"]


def test_japanese_tokens_mark_phrase_starts():
    assert japanese.tokens("日本のいいところ") == [("日本", True), ("の", False), ("いい", True), ("ところ", True)]


def test_chinese_words_and_particles():
    phrases = [p["word"] for p in group_into_phrases(units("你觉得日本好在哪里呢"), "zh")]
    assert phrases == ["你", "觉得", "日本", "好", "在", "哪里呢"]
    assert ("的", False) in chinese.tokens("日本的看法")


def test_cantonese_uses_the_chinese_grouper():
    phrases = [p["word"] for p in group_into_phrases(units("我哋今日問下大家點睇日本"), "yue")]
    assert "".join(phrases) == "我哋今日問下大家點睇日本" and "今日" in phrases


def test_phrase_times_scores_and_attached_punctuation():
    chars = units("です") + [{"word": "ね。", "start": 0.4, "end": 0.6, "score": 0.5, "block": 0}]
    (phrase,) = group_into_phrases(chars, "ja")
    assert phrase == {"word": "ですね。", "start": 0.0, "end": 0.6, "score": (0.9 + 0.9 + 0.5) / 3, "block": 0}


def test_phrases_never_cross_transcript_blocks():
    phrases = group_into_phrases(units("日本", block=0) + units("の", start=1.0, block=1), "ja")
    assert [(p["word"], p["block"]) for p in phrases] == [("日本", 0), ("の", 1)]


def test_languages_aligned_per_word_are_unchanged():
    words = [{"word": "Hello", "start": 0, "end": 1, "score": 1, "block": 0}]
    assert group_into_phrases(words, "en") is words
