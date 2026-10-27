import json
import os
import shutil
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

TRIM_TOP_DB = 40      # quieter than peak - 40 dB counts as silence at the clip edges
EDGE_KEEP = 0.03      # seconds of silence kept at each edge so word onsets/endings are not clipped


# === Step 6: Fit every dubbed clip into its time slot ===
def fit_clips(clips_doc: dict, output_dir: str, video_duration: float | None = None,
              min_speed: float = 0.9, max_speed: float = 1.25, hard_max_speed: float = 1.6,
              max_borrow: float = 0.5, min_gap: float = 0.1, max_retries: int = 2,
              regenerate: Callable[[dict, int, float], tuple[np.ndarray, int, str] | None] | None = None) -> dict:
    """
    Make each clip end before the next one starts, keeping speech natural. For each clip, in order:
      1. trim silence at its edges
      2. if it is shorter than its slot, slow it down (not below min_speed) to fill more of the slot
      3. if it is longer, speed it up (up to max_speed), first into the slot, then into the slot plus
         up to max_borrow seconds of the silence after it (always leaving min_gap before the next line)
      4. if it still does not fit, ask regenerate(clip, max_chars, window) for a shorter line, up to max_retries
      5. last resort: speed it up as much as needed, with a warning above hard_max_speed
    Clips keep their start time, so they stay in sync with the speaker on screen. Lines the TTS could not
    say (status fallback_original / fallback_silence) get no dubbed audio; the status is passed on to the mix.
    Writes <output_dir>/seg_0000.wav and timing_report.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    clips = clips_doc["clips"]
    report = []
    for i, clip in enumerate(clips):
        next_start = clips[i + 1]["start"] if i + 1 < len(clips) else video_duration
        slot = clip["duration"]
        free = (next_start - clip["start"] - slot - min_gap) if next_start is not None else max_borrow
        window = slot + max(0.0, min(max_borrow, free))
        status = clip.get("status", "ok")
        if status.startswith("fallback_"):
            # the original speech keeps its own slot; a silent line takes no time
            end = clip["start"] + (slot if status == "fallback_original" else 0.0)
            report.append({"id": clip["id"], "speaker": clip["speaker"], "start": clip["start"],
                           "end": round(end, 3), "slot": slot, "window": round(window, 3), "status": status,
                           "text": clip["text"], "audio": None, "warning": None})
            continue

        wav, sr = sf.read(clip["audio"], dtype="float32")
        text, retries = clip["text"], 0
        while True:
            wav = trim_silence(wav, sr)
            length = len(wav) / sr
            speed, target = _plan(length, slot, window, min_speed, max_speed)
            if speed <= max_speed or regenerate is None or retries >= max_retries:
                break
            # too long even at max_speed in the full window: ask for a shorter line in proportion
            max_chars = int(len(text) * window * max_speed / length * 0.95)
            result = regenerate(clip, max_chars, window)
            if result is None:
                break
            wav, sr, text = result
            retries += 1
            print(f"✂️ Segment {clip['id']}: shortened to {text!r} ({len(text)} chars)")

        warning = None
        if speed > hard_max_speed:
            warning = f"needs {speed:.2f}x speed-up to fit, above the {hard_max_speed}x limit"
            print(f"⚠️ Segment {clip['id']} {warning}")
        if abs(speed - 1) > 0.01:
            wav = time_stretch(wav, sr, speed)
        # stretching is not sample-exact: never let the clip run past its window
        wav = wav[:int(window * sr)]

        path = os.path.join(output_dir, f"seg_{clip['id']:04d}.wav")
        sf.write(path, wav, sr)
        fitted = len(wav) / sr
        report.append({
            "id": clip["id"],
            "speaker": clip["speaker"],
            "start": clip["start"],
            "status": status,
            "end": round(clip["start"] + fitted, 3),
            "slot": slot,
            "window": round(window, 3),
            "generated": clip["clip_duration"],
            "trimmed": round(length, 3),
            "fitted": round(fitted, 3),
            "coverage": round(fitted / slot, 3),
            "speed": round(speed, 3),
            "gap_used": round(max(0.0, fitted - slot), 3),
            "retranslations": retries,
            "text": text,
            "audio": os.path.abspath(path),
            "warning": warning,
        })

    overlaps = [(a["id"], b["id"]) for a, b in zip(report, report[1:]) if a["end"] > b["start"]]
    document = {"min_speed": min_speed, "max_speed": max_speed, "hard_max_speed": hard_max_speed,
                "max_borrow": max_borrow, "min_gap": min_gap, "overlaps": overlaps, "clips": report}
    with open(os.path.join(output_dir, "timing_report.json"), "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    fitted = [r for r in report if r["audio"]]
    stretched = sum(abs(r["speed"] - 1) > 0.01 for r in fitted)
    borrowed = sum(r["gap_used"] > 0 for r in fitted)
    fallbacks = len(report) - len(fitted)
    measured = [r["coverage"] for r in fitted if r["slot"] >= 1.0]
    coverage = f", speech covers {sum(measured) / len(measured):.0%} of slots ≥ 1 s on average" if measured else ""
    print(f"✅ Fitted {len(fitted)} clips into {output_dir} ({stretched} time-stretched, {borrowed} using the "
          f"following silence, {sum(r['retranslations'] for r in fitted)} retranslations, "
          f"{len(overlaps)} overlaps{f', {fallbacks} not dubbed' if fallbacks else ''}){coverage}")
    return document


def _plan(length: float, slot: float, window: float, min_speed: float, max_speed: float) -> tuple[float, float]:
    """Pick the playback speed (>1 = faster) and the target length for a clip of the given length."""
    if length <= slot:
        target = min(slot, length / min_speed)   # slow down a little to fill the slot, never below min_speed
    elif length / slot <= max_speed:
        target = slot                            # speed up within the slot
    elif length <= window:
        target = length                          # fits as is by using the silence after it
    else:
        target = window                          # speed up into the slot plus the silence after it
    return length / target, target


def trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    _, (start, end) = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    keep = int(EDGE_KEEP * sr)
    return wav[max(0, start - keep):min(len(wav), end + keep)]


def time_stretch(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change speed without changing pitch. Uses Rubber Band if installed, else librosa's phase vocoder."""
    if shutil.which("rubberband"):
        import pyrubberband
        return pyrubberband.time_stretch(wav, sr, speed).astype(np.float32)
    return librosa.effects.time_stretch(wav, rate=speed).astype(np.float32)
