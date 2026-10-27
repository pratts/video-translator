import importlib
import json
import os
import re
from difflib import SequenceMatcher
from functools import cache

import librosa
import numpy as np
import soundfile as sf

SYNTHESIZERS = {"f5": "synthesizers.f5"}
ASR_SAMPLE_RATE = 16000
# a clip is accepted when its transcript is this close to the text it should say (word error rate)...
MAX_WER = 0.2
# ...and no content word (at least this many letters) is missing or heard as a different word
CONTENT_WORD_LENGTH = 4
SIMILAR_WORD = 0.6     # spelling similarity above which a heard word counts as the same word ("Kumagae")
# what to do with a line the TTS cannot say, even after rephrasing
ON_MISMATCH = ("auto", "original", "silence", "keep")
SHORT_LINE = 1.0       # auto: lines up to this long keep the original speaker (often backchannels like はい)


def get_synthesizer(name: str):
    """Import the TTS backend on demand, so unused backends need no packages or models."""
    name = name.lower()
    if name not in SYNTHESIZERS:
        raise ValueError(f"Unknown TTS backend: {name}. Use one of: {', '.join(SYNTHESIZERS)}.")
    return importlib.import_module(SYNTHESIZERS[name])


# === Step 5b: Speak every translated segment in its speaker's cloned voice ===
def synthesize_segments(translation_doc: dict, references: dict, output_dir: str, backend: str = "f5",
                        verify: bool = True, attempts: int = 3, asr_model: str = "large-v3-turbo",
                        on_mismatch: str = "auto", translation_path: str | None = None) -> dict:
    """
    Generate one clip per segment, <output_dir>/seg_0000.wav, sized to the segment's time slot.
    With verify, each clip is transcribed and regenerated with another seed (up to attempts times) when it
    does not say the intended text. If no attempt does, an LLM translator rephrases the line once (saved to
    translation_path) and it is tried again. A line that still fails, or fails to generate at all, gets the
    on_mismatch fallback:
      original: later stages put the original speaker's audio back for that segment ("fallback_original")
      silence:  the line is dropped ("fallback_silence")
      keep:     the closest attempt is used anyway ("mismatch")
      auto:     original for lines up to SHORT_LINE seconds, silence for longer ones
    Saves clips.json describing every clip.
    """
    if on_mismatch not in ON_MISMATCH:
        raise ValueError(f"Unknown TTS_ON_MISMATCH: {on_mismatch}. Use one of: {', '.join(ON_MISMATCH)}.")
    synthesizer = get_synthesizer(backend)
    language = translation_doc["target_language"]
    rephrase = _rephraser(translation_doc, translation_path) if verify else None
    os.makedirs(output_dir, exist_ok=True)

    clips = []
    for seg in translation_doc["segments"]:
        path = os.path.join(output_dir, f"seg_{seg['id']:04d}.wav")
        clip = {"id": seg["id"], "speaker": seg["speaker"], "start": seg["start"], "duration": seg["duration"],
                "gap_after": seg["gap_after"], "text": seg["translation"], "audio": os.path.abspath(path)}
        print(f"🗣️ [{seg['id'] + 1}/{len(translation_doc['segments'])}] {seg['speaker']}: {seg['translation']}")
        try:
            result = _generate(synthesizer, seg["translation"], references[seg["speaker"]], language,
                               seg["duration"], verify, attempts, asr_model)
            if verify and not result["ok"] and rephrase:
                print(f"🔁 Segment {seg['id']} still sounds like {result['heard']!r}; rephrasing the line")
                text = rephrase(seg["id"], result["heard"])
                retry = _generate(synthesizer, text, references[seg["speaker"]], language, seg["duration"],
                                  verify, attempts, asr_model)
                retry["attempts"] += result["attempts"]
                clip.update(rephrased_from=clip["text"], text=text)
                result = retry
            wav, sr = result["wav"], result["sr"]
            status = "unverified" if not verify else "ok" if result["ok"] else _fallback(on_mismatch, seg["duration"])
            if status != "ok" and status != "unverified":
                print(f"⚠️ Segment {seg['id']} still sounds like {result['heard']!r} after {result['attempts']} "
                      f"attempts; {_FALLBACK_TEXT[status]}")
            clip.update(heard=result["heard"], wer=None if result["wer"] is None else round(result["wer"], 3),
                        missed=result["missed"], attempts=result["attempts"])
        except Exception as e:  # keep going: one bad line should not stop the whole dub
            sr = synthesizer.SAMPLE_RATE
            wav = np.zeros(int(seg["duration"] * sr), dtype=np.float32)
            status = _fallback(on_mismatch, seg["duration"])
            if status == "mismatch":
                status = "failed"     # nothing to keep: the silent clip stands in
            print(f"❌ Segment {seg['id']} failed ({e}); {_FALLBACK_TEXT[status]}")
            clip["error"] = str(e)

        sf.write(path, wav, sr)
        clip.update(status=status, clip_duration=round(len(wav) / sr, 3))
        clips.append(clip)

    document = {"backend": backend, "language": language, "on_mismatch": on_mismatch, "clips": clips}
    with open(os.path.join(output_dir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    counts = {s: sum(c["status"] == s for c in clips) for s in STATUSES}
    print(f"✅ Saved {len(clips)} clips to {output_dir} ({', '.join(f'{v} {k}' for k, v in counts.items() if v)})")
    return document


STATUSES = ("ok", "unverified", "mismatch", "fallback_original", "fallback_silence", "failed")
_FALLBACK_TEXT = {"mismatch": "keeping the closest attempt", "fallback_original": "keeping the original speech",
                  "fallback_silence": "leaving the line silent", "failed": "writing silence"}


def _fallback(policy: str, duration: float) -> str:
    """Status for a line the TTS could not say, from the TTS_ON_MISMATCH policy."""
    if policy == "auto":
        policy = "original" if duration <= SHORT_LINE else "silence"
    return {"original": "fallback_original", "silence": "fallback_silence", "keep": "mismatch"}[policy]


def fallback_summary(clips_doc: dict) -> list[str]:
    """One line per segment that is not dubbed as written (for the end-of-run summary)."""
    lines = []
    for c in clips_doc["clips"]:
        if c["status"] in ("mismatch", "fallback_original", "fallback_silence", "failed"):
            heard = f", heard {c['heard']!r}" if c.get("heard") is not None else ""
            lines.append(f"segment {c['id']} ({c['start']:.2f}s, {c['speaker']}): {c['status']} — "
                         f"{c['text']!r}{heard}")
        elif c.get("rephrased_from"):
            lines.append(f"segment {c['id']} ({c['start']:.2f}s, {c['speaker']}): rephrased "
                         f"{c['rephrased_from']!r} → {c['text']!r}")
    return lines


def _generate(synthesizer, text: str, reference: dict, language: str, duration: float | None, verify: bool,
              attempts: int, asr_model: str) -> dict:
    """
    Synthesize text, retrying with new seeds until speech recognition hears it.
    Returns the closest attempt: {wav, sr, heard, wer, missed, ok, attempts}.
    """
    best, tries = None, 0
    for seed in range(attempts if verify else 1):
        tries += 1
        wav, sr = synthesizer.synthesize(text, reference, language, target_duration=duration, seed=seed)
        result = {"wav": wav, "sr": sr, "heard": None, "wer": None, "missed": [], "ok": True}
        if verify:
            heard = _transcribe(wav, sr, language, asr_model)
            error, missed = word_error_rate(text, heard), missed_words(text, heard)
            result.update(heard=heard, wer=error, missed=missed, ok=error <= MAX_WER and not missed)
        if best is None or (len(result["missed"]), result["wer"] or 0) < (len(best["missed"]), best["wer"] or 0):
            best = result
        if result["ok"]:
            break
    return {**best, "attempts": tries}


def _rephraser(translation_doc: dict, translation_path: str | None):
    """rephrase(segment_id, heard) -> new line, saved to translation_path; None if the translator can't."""
    from utils.translate import get_translator, rephrase, save_translation

    if not translation_path or not hasattr(get_translator(translation_doc["translator"]), "rephrase"):
        return None
    folder = os.path.dirname(translation_path)

    def run(segment_id: int, heard: str) -> str:
        text = rephrase(translation_doc, segment_id, heard)
        save_translation(translation_doc, translation_path, os.path.join(folder, "translated_raw.srt"),
                         os.path.join(folder, "translated_polished.srt"))
        return text

    return run


def make_regenerator(translation_doc: dict, translation_path: str, references: dict, backend: str = "f5",
                     verify: bool = True, attempts: int = 3, asr_model: str = "large-v3-turbo", **_):
    """
    Build the callback the timing-fit step uses when a clip is too long: shorten the line's translation
    (LLM translators only) and synthesize it again for the available time. Returns None when the
    translator cannot shorten lines. The shortened translation is saved back to translation_path; if the
    shortened line can't be synthesized so that it is heard correctly, the callback returns None and the
    translation is left as it was.
    """
    from utils.translate import get_translator, save_translation, shorten

    if not hasattr(get_translator(translation_doc["translator"]), "shorten"):
        return None
    synthesizer = get_synthesizer(backend)
    language = translation_doc["target_language"]
    folder = os.path.dirname(translation_path)

    def save():
        save_translation(translation_doc, translation_path, os.path.join(folder, "translated_raw.srt"),
                         os.path.join(folder, "translated_polished.srt"))

    def regenerate(clip: dict, max_chars: int, window: float):
        seg = next(s for s in translation_doc["segments"] if s["id"] == clip["id"])
        before = {k: seg[k] for k in ("translation", "max_chars", "fits")}
        text = shorten(translation_doc, clip["id"], max_chars)
        result = _generate(synthesizer, text, references[clip["speaker"]], language,
                           window if synthesizer.SUPPORTS_DURATION else None, verify, attempts, asr_model)
        if not result["ok"]:
            print(f"⚠️ Segment {clip['id']}: shortened line {text!r} sounds like {result['heard']!r}; "
                  f"keeping the longer one")
            seg.update(before)
            return None
        save()
        return result["wav"], result["sr"], text

    return regenerate


@cache
def _asr(model_name: str, language: str):
    import whisperx
    return whisperx.load_model(model_name, "cpu", compute_type="int8", language=language)


def _transcribe(wav: np.ndarray, sr: int, language: str, model_name: str) -> str:
    audio = librosa.resample(wav, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
    result = _asr(model_name, language).transcribe(audio, batch_size=4, language=language)
    return " ".join(s["text"].strip() for s in result["segments"]).strip()


def _words(text: str) -> list[str]:
    # split joined names ("EasyJapanese" is spoken, and transcribed, as "Easy Japanese")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return re.sub(r"[^\w' ]", " ", text.lower()).split()


def missed_words(expected: str, heard: str) -> list[str]:
    """
    Content words of expected that were not heard: missing, or heard as a clearly different word
    ("strengths" heard as "drinks"). Short words (articles, fillers) and near-spellings of names are ignored.
    """
    hyp = _words(heard)
    joined = "".join(hyp)
    return [w for w in _words(expected) if len(w) >= CONTENT_WORD_LENGTH and w not in joined
            and not any(SequenceMatcher(None, w, h).ratio() >= SIMILAR_WORD for h in hyp)]


def word_error_rate(expected: str, heard: str) -> float:
    """Edit distance between word sequences, divided by the number of expected words."""
    ref, hyp = _words(expected), _words(heard)
    if "".join(ref) == "".join(hyp):  # same words, spaced differently ("Mana Kumagai" heard as "Manakumagai")
        return 0.0
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ref[i - 1] != hyp[j - 1]))
    return row[len(hyp)] / max(len(ref), 1)
