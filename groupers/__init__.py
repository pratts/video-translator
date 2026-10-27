"""
Phrase groupers for languages that WhisperX aligns per character (Japanese, Chinese, Cantonese).

A speaker change taken from diarization can fall inside a word when every character is its own unit
(け|ど, 日|本). A grouper merges the characters of each transcript block into phrases that nobody would
switch speaker in the middle of, so speakers are assigned and segments split per phrase.

Each language module provides tokens(text) -> [(surface, starts_phrase)]: the text split into words,
each marked with whether it starts a new phrase or attaches to the phrase before it.
Languages aligned per word (with spaces) need no grouper.
"""
import importlib
import unicodedata

GROUPERS = {"ja": "groupers.japanese", "zh": "groupers.chinese", "yue": "groupers.chinese"}


def group_into_phrases(units: list[dict], language: str) -> list[dict]:
    """Merge aligned units {"word", "start", "end", "score", "block"} into phrases; unchanged if not needed."""
    if language not in GROUPERS:
        return units
    tokenizer = importlib.import_module(GROUPERS[language])
    return [phrase for block in _blocks(units) for phrase in _group_block(block, tokenizer.tokens)]


def _group_block(units: list[dict], tokens) -> list[dict]:
    text = "".join(u["word"] for u in units)
    owner, pos, phrase = [], 0, -1     # owner[i] = phrase index of text[i]
    for surface, starts_phrase in tokens(text):
        start = text.find(surface, pos)
        if start < 0:                  # the tokenizer normalized something: keep the rest in the current phrase
            break
        owner += [max(phrase, 0)] * (start - pos)
        if phrase < 0 or (starts_phrase and not _is_punctuation(surface)):
            phrase += 1
        owner += [phrase] * len(surface)
        pos = start + len(surface)
    owner += [max(phrase, 0)] * (len(text) - pos)

    # a unit belongs to the phrase of its first character (units can carry attached punctuation, e.g. す。)
    groups, offset = {}, 0
    for u in units:
        groups.setdefault(owner[offset], []).append(u)
        offset += len(u["word"])
    return [{"word": "".join(u["word"] for u in members),
             "start": members[0]["start"],
             "end": members[-1]["end"],
             "score": _mean([u.get("score") for u in members]),
             "block": members[0]["block"]}
            for members in groups.values()]


def _blocks(units: list[dict]) -> list[list[dict]]:
    blocks = []
    for u in units:
        if not blocks or blocks[-1][-1]["block"] != u["block"]:
            blocks.append([])
        blocks[-1].append(u)
    return blocks


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(c).startswith("P") for c in text)


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
