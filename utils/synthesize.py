import importlib
import json
import os
import re
from functools import cache

import librosa
import numpy as np
import soundfile as sf

SYNTHESIZERS = {"f5": "synthesizers.f5"}
ASR_SAMPLE_RATE = 16000
# a clip is accepted when its transcript is this close to the text it should say (word error rate)
MAX_WER = 0.2


def get_synthesizer(name: str):
    """Import the TTS backend on demand, so unused backends need no packages or models."""
    name = name.lower()
    if name not in SYNTHESIZERS:
        raise ValueError(f"Unknown TTS backend: {name}. Use one of: {', '.join(SYNTHESIZERS)}.")
    return importlib.import_module(SYNTHESIZERS[name])


# === Step 5b: Speak every translated segment in its speaker's cloned voice ===
def synthesize_segments(translation_doc: dict, references: dict, output_dir: str, backend: str = "f5",
                        verify: bool = True, attempts: int = 3, asr_model: str = "large-v3-turbo") -> list[dict]:
    """
    Generate one clip per segment, <output_dir>/seg_0000.wav, sized to the segment's time slot.
    With verify, each clip is transcribed and regenerated with another seed (up to attempts times) when it
    does not say the intended text; the closest attempt is kept. A segment that fails to generate becomes
    silence, so later stages still get one clip per segment. Saves clips.json describing every clip.
    """
    synthesizer = get_synthesizer(backend)
    language = translation_doc["target_language"]
    os.makedirs(output_dir, exist_ok=True)

    clips = []
    for seg in translation_doc["segments"]:
        path = os.path.join(output_dir, f"seg_{seg['id']:04d}.wav")
        clip = {"id": seg["id"], "speaker": seg["speaker"], "start": seg["start"], "duration": seg["duration"],
                "gap_after": seg["gap_after"], "text": seg["translation"], "audio": os.path.abspath(path)}
        print(f"🗣️ [{seg['id'] + 1}/{len(translation_doc['segments'])}] {seg['speaker']}: {seg['translation']}")
        try:
            best, tries = None, 0
            for seed in range(attempts if verify else 1):
                tries += 1
                wav, sr = synthesizer.synthesize(seg["translation"], references[seg["speaker"]], language,
                                                 target_duration=seg["duration"], seed=seed)
                heard = _transcribe(wav, sr, language, asr_model) if verify else None
                error = word_error_rate(seg["translation"], heard) if verify else None
                if best is None or (verify and error < best[3]):
                    best = (wav, sr, heard, error)
                if not verify or error <= MAX_WER:
                    break
            wav, sr, heard, error = best
            status = "unverified" if not verify else "ok" if error <= MAX_WER else "mismatch"
            if status == "mismatch":
                print(f"⚠️ Segment {seg['id']} still sounds like {heard!r} after {tries} attempts; keeping the closest")
            clip.update(heard=heard, wer=None if error is None else round(error, 3), attempts=tries)
        except Exception as e:  # keep going: one bad line should not stop the whole dub
            print(f"❌ Segment {seg['id']} failed ({e}); writing silence")
            sr = synthesizer.SAMPLE_RATE
            wav = np.zeros(int(seg["duration"] * sr), dtype=np.float32)
            status = "failed"
            clip["error"] = str(e)

        sf.write(path, wav, sr)
        clip.update(status=status, clip_duration=round(len(wav) / sr, 3))
        clips.append(clip)

    with open(os.path.join(output_dir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": backend, "language": language, "clips": clips}, f, ensure_ascii=False, indent=2)
    counts = {s: sum(c["status"] == s for c in clips) for s in ("ok", "mismatch", "failed", "unverified")}
    print(f"✅ Saved {len(clips)} clips to {output_dir} ({', '.join(f'{v} {k}' for k, v in counts.items() if v)})")
    return clips


@cache
def _asr(model_name: str, language: str):
    import whisperx
    return whisperx.load_model(model_name, "cpu", compute_type="int8", language=language)


def _transcribe(wav: np.ndarray, sr: int, language: str, model_name: str) -> str:
    audio = librosa.resample(wav, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
    result = _asr(model_name, language).transcribe(audio, batch_size=4, language=language)
    return " ".join(s["text"].strip() for s in result["segments"]).strip()


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w' ]", " ", text.lower()).split()


def word_error_rate(expected: str, heard: str) -> float:
    """Edit distance between word sequences, divided by the number of expected words."""
    ref, hyp = _words(expected), _words(heard)
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ref[i - 1] != hyp[j - 1]))
    return row[len(hyp)] / max(len(ref), 1)
