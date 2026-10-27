import json
import os
import numpy as np
import soundfile as sf
import librosa
from utils.segment import SENTENCE_END

REFERENCE_SAMPLE_RATE = 24000
EDGE_PAD = 0.05     # seconds kept around each segment so word edges are not cut
JOIN_GAP = 0.25     # silence between joined pieces
MIN_REFERENCE = 5.0  # F5-TTS clones a voice noticeably less well from a shorter reference


# === Step 5a: Pick a clean voice reference for each speaker ===
def build_voice_references(segments_doc: dict, turns: list[dict], vocals_path: str, background_path: str,
                           output_dir: str, target_duration: float = 8.0, max_duration: float = 11.0,
                           min_segment: float = 1.0, manual_dir: str | None = None,
                           asr_model: str = "large-v3-turbo") -> dict:
    """
    For each speaker, join their cleanest segments into one reference clip of about target_duration seconds
    (F5-TTS clips references at 12 s). Segments are ranked by alignment confidence, separation quality
    (vocals vs. background level), overlap with other speakers and whether they are complete sentences:
    a reference that stops mid-sentence makes the TTS continue it at the start of every line. Segments shorter
    than min_segment are only used when the speaker has too little other speech.
    A manual reference, <manual_dir>/<speaker>.wav with its transcript in <speaker>.txt (transcribed with
    Whisper when missing), replaces the automatic one. Saves <output_dir>/<speaker>.wav and
    references.json: {speaker: {"audio", "text", "language", "duration", "segments", "source"}}.
    """
    vocals, sr = _load_mono(vocals_path)
    background, _ = _load_mono(background_path, sr)
    language = segments_doc["language"]
    joiner = "" if language in ("ja", "zh", "yue") else " "

    os.makedirs(output_dir, exist_ok=True)
    references = {}
    for speaker in segments_doc["speakers"]:
        manual = _manual_reference(manual_dir, speaker, output_dir, asr_model, language)
        if manual:
            references[speaker] = manual
            print(f"🎤 {speaker}: {manual['duration']:.1f}s manual reference from {manual['source_file']}")
            continue

        own = [s for s in segments_doc["segments"] if s["speaker"] == speaker]
        quality = {s["id"]: _quality(s, turns, vocals, background, sr) for s in own}
        rank = lambda segs: sorted(segs, key=lambda s: quality[s["id"]], reverse=True)
        chosen, total = _pick(rank([s for s in own if s["duration"] >= min_segment]), [], 0.0,
                              target_duration, max_duration)
        if total < MIN_REFERENCE:  # too little: short lines are better than a too-short reference
            chosen, total = _pick(rank([s for s in own if s not in chosen]), chosen, total,
                                  target_duration, max_duration)
        # keep time order, but end on a complete sentence if possible
        chosen.sort(key=lambda s: s["start"])
        complete = [s for s in chosen if _complete(s)]
        if complete and not _complete(chosen[-1]):
            chosen.remove(complete[-1])
            chosen.append(complete[-1])

        gap = np.zeros(int(JOIN_GAP * sr), dtype=np.float32)
        pieces = []
        for seg in chosen:
            start = max(0, int((seg["start"] - EDGE_PAD) * sr))
            end = min(len(vocals), int((seg["end"] + EDGE_PAD) * sr))
            pieces += [vocals[start:end], gap]
        audio = np.concatenate(pieces[:-1])
        audio = librosa.resample(audio, orig_sr=sr, target_sr=REFERENCE_SAMPLE_RATE)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (0.9 / peak)

        path = os.path.join(output_dir, f"{speaker}.wav")
        sf.write(path, audio, REFERENCE_SAMPLE_RATE)
        references[speaker] = {
            "audio": os.path.abspath(path),
            "text": joiner.join(s["text"] for s in chosen),
            "language": language,
            "duration": round(len(audio) / REFERENCE_SAMPLE_RATE, 3),
            "segments": [s["id"] for s in chosen],
            "source": "auto",
        }
        print(f"🎤 {speaker}: {references[speaker]['duration']:.1f}s reference from segments "
              f"{references[speaker]['segments']}")
        if references[speaker]["duration"] < MIN_REFERENCE:
            spoken = sum(s["duration"] for s in own)
            print(f"⚠️ Only {references[speaker]['duration']:.1f}s of clean speech for {speaker} "
                  f"({spoken:.1f}s in all {len(own)} of their lines); the cloned voice may be less similar. "
                  f"For a better voice, put a 5-12 s clip of them in voice_references_dir as {speaker}.wav")

    with open(os.path.join(output_dir, "references.json"), "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)
    return references


def _pick(ranked: list[dict], chosen: list[dict], total: float, target_duration: float,
          max_duration: float) -> tuple[list[dict], float]:
    """Add the best segments until the reference reaches target_duration, never going over max_duration."""
    chosen = list(chosen)
    for seg in ranked:
        length = seg["duration"] + 2 * EDGE_PAD + (JOIN_GAP if chosen else 0)
        if chosen and total + length > max_duration:
            continue
        chosen.append(seg)
        total += length
        if total >= target_duration:
            break
    return chosen, total


def _manual_reference(manual_dir: str | None, speaker: str, output_dir: str, asr_model: str,
                      language: str) -> dict | None:
    """
    The speaker's reference from manual_dir, resampled and normalized like automatic ones, or None. A .txt
    transcript is taken to be in the video's language; without one, Whisper detects the language.
    """
    source = os.path.join(manual_dir, f"{speaker}.wav") if manual_dir else None
    if not source or not os.path.exists(source):
        return None
    audio, sr = _load_mono(source)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=REFERENCE_SAMPLE_RATE) if sr != REFERENCE_SAMPLE_RATE \
        else audio
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (0.9 / peak)
    path = os.path.join(output_dir, f"{speaker}.wav")
    sf.write(path, audio, REFERENCE_SAMPLE_RATE)

    transcript = os.path.splitext(source)[0] + ".txt"
    if os.path.exists(transcript):
        with open(transcript, encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text, language = _transcribe(audio, asr_model)
    duration = len(audio) / REFERENCE_SAMPLE_RATE
    if duration > 12:
        print(f"⚠️ {source} is {duration:.1f}s; F5-TTS only uses the first 12 s, so its transcript will not match")
    return {"audio": os.path.abspath(path), "text": text, "language": language, "duration": round(duration, 3),
            "segments": [], "source": "manual", "source_file": os.path.abspath(source)}


def _transcribe(audio: np.ndarray, model_name: str) -> tuple[str, str]:
    """Transcript and detected language of a manual reference that came without a .txt file."""
    import whisperx
    model = whisperx.load_model(model_name, "cpu", compute_type="int8")
    result = model.transcribe(librosa.resample(audio, orig_sr=REFERENCE_SAMPLE_RATE, target_sr=16000), batch_size=4)
    return " ".join(s["text"].strip() for s in result["segments"]).strip(), result["language"]


def _load_mono(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr and rate != sr:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
        rate = sr
    return audio, rate


def _complete(seg: dict) -> bool:
    return seg["text"].strip().endswith(SENTENCE_END)


def _quality(seg: dict, turns: list[dict], vocals: np.ndarray, background: np.ndarray, sr: int) -> float:
    """Higher is cleaner: confident alignment, loud vocals over background, no other speaker talking."""
    scores = [w["score"] for w in seg["words"] if w.get("score") is not None]
    confidence = float(np.mean(scores)) if scores else 0.5

    start, end = int(seg["start"] * sr), int(seg["end"] * sr)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) + 1e-8
    snr = 20 * np.log10(rms(vocals[start:end]) / rms(background[start:end]))

    overlap = sum(max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
                  for t in turns if t["speaker"] != seg["speaker"])

    # snr is clipped to 0–30 dB; overlap is the fraction of the segment with another speaker talking
    return confidence + min(max(snr, 0), 30) / 30 - 2 * overlap / max(seg["duration"], 1e-3) \
        + min(seg["duration"], 6) / 12 + (0.5 if _complete(seg) else 0)
