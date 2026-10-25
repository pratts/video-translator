import json
import os
import srt
import unicodedata
from datetime import timedelta
from statistics import median

# Languages whose words are aligned per character and joined without spaces
NO_SPACE_LANGUAGES = {"ja", "zh", "yue"}
SENTENCE_END = ("。", "！", "？", "!", "?", ".", "…")
CLAUSE_END = ("、", "，", ",", ";", "；", ":", "：")


# === Step 3b: Build dubbing segments from timed words and speaker turns ===
def build_segments(transcript: dict, turns: list[dict], video_duration: float | None = None,
                   min_turn: float = 0.5, pause: float = 0.35, min_duration: float = 1.0,
                   max_duration: float = 12.0, merge_gap: float = 0.6) -> dict:
    """
    Turn word-level transcript + diarization into sentence-level, single-speaker dubbing segments.
    - min_turn: speaker runs shorter than this (s) are treated as diarization noise and absorbed by a neighbour
    - pause: silence (s) between words that ends a segment
    - min_duration / merge_gap: segments shorter than min_duration are merged into a same-speaker
      neighbour that is at most merge_gap away
    - max_duration: longer segments are split at their best internal pause or clause boundary
    """
    no_spaces = transcript["language"] in NO_SPACE_LANGUAGES
    words = _flatten_words(transcript)
    if not words:
        return _document(transcript, [], video_duration, no_spaces)

    _trim_long_words(words, no_spaces)
    _assign_speakers(words, turns)
    _smooth_speakers(words, min_turn)

    segments = _split(words, pause)
    segments = _merge_short(segments, min_duration, max_duration, merge_gap)
    segments = [part for seg in segments for part in _split_long(seg, max_duration, min_duration)]

    return _document(transcript, segments, video_duration, no_spaces)


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(c).startswith("P") for c in text)


def _flatten_words(transcript: dict) -> list[dict]:
    """All words in order, tagged with their transcript block. Punctuation is attached to the word before it."""
    words = []
    for block in transcript["segments"]:
        for w in block["words"]:
            text = w["word"].strip()
            if not text:
                continue
            if _is_punctuation(text) and words and words[-1]["block"] == block["id"]:
                words[-1]["word"] += text
                continue
            words.append({**w, "block": block["id"]})
    return words


def _is_ideograph(text: str) -> bool:
    return any("\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" for c in text)


def _trim_long_words(words: list[dict], no_spaces: bool):
    """
    Forced alignment leaves no gaps between words, so a pause is absorbed into a neighbouring word
    (e.g. a Japanese character lasting 1.8 s). Shorten such words so the pause shows up as a gap.
    The pause is before the word if it starts a block or follows a sentence/clause end, otherwise after it.
    """
    typical = median(w["end"] - w["start"] for w in words)
    short = max(typical * 2, 0.2)
    for i, w in enumerate(words):
        # one kanji can carry several syllables (魚 = sa-ka-na), so allow it more time than kana or letters
        threshold = (1.0 if _is_ideograph(w["word"]) else 0.5) if no_spaces else 1.2
        if w["end"] - w["start"] <= max(threshold, typical * 4):
            continue
        prev = words[i - 1] if i > 0 and words[i - 1]["block"] == w["block"] else None
        if prev is None or prev["word"].strip().endswith(SENTENCE_END + CLAUSE_END):
            w["start"] = w["end"] - short
        else:
            w["end"] = w["start"] + short


def _assign_speakers(words: list[dict], turns: list[dict]):
    """Give each word the speaker it overlaps most, or the nearest speaker turn if none overlaps."""
    for w in words:
        if not turns:
            w["speaker"] = "SPEAKER_00"
            continue
        overlap = {}
        for t in turns:
            o = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if o > 0:
                overlap[t["speaker"]] = overlap.get(t["speaker"], 0) + o
        if overlap:
            w["speaker"] = max(overlap, key=overlap.get)
        else:
            mid = (w["start"] + w["end"]) / 2
            w["speaker"] = min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - mid))["speaker"]


def _runs(words: list[dict]) -> list[tuple[int, int]]:
    """Index ranges [start, end) of consecutive words with the same speaker."""
    runs, start = [], 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i]["speaker"] != words[start]["speaker"]:
            runs.append((start, i))
            start = i
    return runs


def _smooth_speakers(words: list[dict], min_turn: float):
    """Absorb very short speaker runs (diarization flicker, mid-word speaker changes) into a neighbour."""
    while True:
        runs = _runs(words)
        if len(runs) < 2:
            return
        duration = lambda r: words[r[1] - 1]["end"] - words[r[0]]["start"]
        idx = min(range(len(runs)), key=lambda k: duration(runs[k]))
        if duration(runs[idx]) >= min_turn:
            return

        start, end = runs[idx]
        prev_run = runs[idx - 1] if idx > 0 else None
        next_run = runs[idx + 1] if idx + 1 < len(runs) else None
        if prev_run and next_run:
            gap_prev = words[start]["start"] - words[prev_run[1] - 1]["end"]
            gap_next = words[next_run[0]]["start"] - words[end - 1]["end"]
            target = words[prev_run[0]] if gap_prev <= gap_next else words[next_run[0]]
        else:
            target = words[(prev_run or next_run)[0]]
        for w in words[start:end]:
            w["speaker"] = target["speaker"]


def _split(words: list[dict], pause: float) -> list[list[dict]]:
    """Cut after sentence ends, pauses, speaker changes and transcript block boundaries."""
    segments, current = [], []
    for i, w in enumerate(words):
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (nxt is None
                or nxt["block"] != w["block"]
                or nxt["speaker"] != w["speaker"]
                or w["word"].strip().endswith(SENTENCE_END)
                or nxt["start"] - w["end"] >= pause):
            segments.append(current)
            current = []
    return segments


def _duration(seg: list[dict]) -> float:
    return seg[-1]["end"] - seg[0]["start"]


def _merge_short(segments: list[list[dict]], min_duration: float, max_duration: float,
                 merge_gap: float) -> list[list[dict]]:
    """Merge too-short segments into the closest same-speaker neighbour."""
    merged = True
    while merged:
        merged = False
        for i, seg in enumerate(segments):
            if _duration(seg) >= min_duration:
                continue
            candidates = []
            if i > 0:
                candidates.append((seg[0]["start"] - segments[i - 1][-1]["end"], i - 1))
            if i + 1 < len(segments):
                candidates.append((segments[i + 1][0]["start"] - seg[-1]["end"], i + 1))
            for gap, j in sorted(candidates):
                other = segments[j]
                first, second = (other, seg) if j < i else (seg, other)
                if (other[0]["speaker"] == seg[0]["speaker"] and gap <= merge_gap
                        and second[-1]["end"] - first[0]["start"] <= max_duration):
                    segments[min(i, j)] = first + second
                    del segments[max(i, j)]
                    merged = True
                    break
            if merged:
                break
    return segments


def _split_long(seg: list[dict], max_duration: float, min_duration: float) -> list[list[dict]]:
    """Split a too-long segment at the internal boundary with the biggest pause, preferring clause ends."""
    if _duration(seg) <= max_duration or len(seg) < 2:
        return [seg]
    middle = (seg[0]["start"] + seg[-1]["end"]) / 2

    def score(k):  # boundary after word k
        gap = seg[k + 1]["start"] - seg[k]["end"]
        clause = 0.3 if seg[k]["word"].strip().endswith(CLAUSE_END) else 0
        centered = 1 - abs(seg[k]["end"] - middle) / (_duration(seg) / 2)
        return gap + clause + 0.1 * centered

    candidates = [k for k in range(len(seg) - 1)
                  if _duration(seg[:k + 1]) >= min_duration and _duration(seg[k + 1:]) >= min_duration]
    if not candidates:
        return [seg]
    k = max(candidates, key=score)
    return _split_long(seg[:k + 1], max_duration, min_duration) + _split_long(seg[k + 1:], max_duration, min_duration)


def _document(transcript: dict, segments: list[list[dict]], video_duration: float | None,
              no_spaces: bool) -> dict:
    out = []
    for i, seg in enumerate(segments):
        start, end = seg[0]["start"], seg[-1]["end"]
        next_start = segments[i + 1][0]["start"] if i + 1 < len(segments) else video_duration
        out.append({
            "id": i,
            "speaker": seg[0]["speaker"],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "gap_after": round(next_start - end, 3) if next_start is not None else None,
            "text": ("" if no_spaces else " ").join(w["word"].strip() for w in seg),
            "words": [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3),
                       "score": w.get("score")} for w in seg],
        })
    return {
        "language": transcript["language"],
        "duration": video_duration,
        "speakers": sorted({s["speaker"] for s in out}),
        "segments": out,
    }


def save_segments(document: dict, json_output_path: str, srt_output_path: str):
    """Save segments as JSON (used by later stages) and as an SRT with speaker labels for review."""
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    subtitles = [srt.Subtitle(index=s["id"] + 1, start=timedelta(seconds=s["start"]),
                              end=timedelta(seconds=s["end"]), content=f"[{s['speaker']}] {s['text']}")
                 for s in document["segments"]]
    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    print(f"✅ Saved {len(document['segments'])} segments ({len(document['speakers'])} speakers) "
          f"to {json_output_path} and {srt_output_path}")
