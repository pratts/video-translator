import json
import os

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

MIN_LOUDNESS_SECONDS = 0.4   # pyloudnorm needs at least one 400 ms block
MAX_GAIN_DB = 12             # never boost or cut a dubbed line by more than this


# === Step 7: Mix the dubbed speech with the original background ===
def mix_dub(timing_report: dict, vocals_path: str, background_path: str, duration: float, output_dir: str,
            target_lufs: float = -16.0, true_peak: float = -1.5, duck_db: float = -6.0, attack: float = 0.08,
            release: float = 0.25, fade: float = 0.01, keep_vocals_db: float | None = None) -> dict:
    """
    Build the dubbed soundtrack, as long as the video, at the background's sample rate and channels:
    - each fitted clip gets short fades and is matched to the loudness of the original speech it replaces
      (per segment when long enough to measure, otherwise the speaker's average)
    - the background stays at its original level, and is lowered by duck_db only while dubbed speech plays,
      with attack/release ramps; sections without speech are untouched
    - lines the TTS could not say keep the original speaker's voice (status fallback_original), unducked and
      at its original level, or stay silent (fallback_silence)
    - optionally, original vocal sounds outside dubbed lines (laughs, breaths) are kept at keep_vocals_db
    - the mix is normalized to target_lufs with one gain; peaks above true_peak dBTP are limited
    Writes <output_dir>/dubbed_audio.wav, speech.wav (dubbed speech only) and mix.json.
    """
    background, sr = _read(background_path, duration)
    vocals, _ = _read(vocals_path, duration, sr)
    length, channels = background.shape
    meter = pyloudnorm.Meter(sr)
    vocals_mono = vocals.mean(axis=1)

    # original speech put back where the dub fell back to it
    restored = np.zeros(length, dtype=np.float32)
    report = []
    for clip in timing_report["clips"]:
        status = clip.get("status", "ok")
        if status.startswith("fallback_"):
            start = int(round(clip["start"] * sr))
            end = min(length, int(round(clip["end"] * sr)))
            if status == "fallback_original" and end > start:
                restored[start:end] = 1.0
            report.append({"id": clip["id"], "speaker": clip["speaker"], "start": clip["start"],
                           "end": clip["end"], "status": status})

    clips = []
    for clip in timing_report["clips"]:
        if not clip.get("audio"):
            continue
        wav, clip_sr = sf.read(clip["audio"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = librosa.resample(wav, orig_sr=clip_sr, target_sr=sr) if clip_sr != sr else wav
        start = int(round(clip["start"] * sr))
        wav = _fade(wav[:max(0, length - start)], int(fade * sr))
        original = vocals_mono[start:start + len(wav)]
        clips.append({"clip": clip, "wav": wav, "start": start,
                      "dub_lufs": _loudness(meter, wav), "original_lufs": _loudness(meter, original)})

    # speaker averages, used for lines too short to measure on their own
    speaker_gain = {}
    for speaker in sorted({c["clip"]["speaker"] for c in clips}):
        own = [c for c in clips if c["clip"]["speaker"] == speaker]
        dub = _loudness(meter, np.concatenate([c["wav"] for c in own]))
        original = _loudness(meter, np.concatenate([vocals_mono[c["start"]:c["start"] + len(c["wav"])] for c in own]))
        speaker_gain[speaker] = float(original - dub) if dub is not None and original is not None else 0.0

    speech = np.zeros(length, dtype=np.float32)
    envelope = np.ones(length, dtype=np.float32)
    duck = 10 ** (duck_db / 20)
    for c in clips:
        clip, wav, start = c["clip"], c["wav"], c["start"]
        if c["dub_lufs"] is not None and c["original_lufs"] is not None:
            gain_db = c["original_lufs"] - c["dub_lufs"]
        else:
            gain_db = speaker_gain[clip["speaker"]]
        gain_db = float(np.clip(gain_db, -MAX_GAIN_DB, MAX_GAIN_DB))
        speech[start:start + len(wav)] += wav * 10 ** (gain_db / 20)
        _duck(envelope, start, start + len(wav), int(attack * sr), int(release * sr), duck)
        report.append({"id": clip["id"], "speaker": clip["speaker"], "start": clip["start"],
                       "end": round((start + len(wav)) / sr, 3), "status": clip.get("status", "ok"),
                       "gain_db": round(gain_db, 2),
                       "original_lufs": _round(c["original_lufs"]), "dub_lufs": _round(c["dub_lufs"])})

    # short ramps at the edges of restored speech, so it does not click
    restored = uniform_filter1d(restored, max(1, int(fade * sr)))
    mix = background * envelope[:, None] + speech[:, None] + vocals * restored[:, None]
    if keep_vocals_db is not None:
        # 0 under dubbed lines (same ramps as the ducking) and restored original speech, 1 elsewhere
        outside = (envelope - duck) / (1 - duck) if duck < 1 else (envelope >= 1).astype(np.float32)
        mix += vocals * (np.minimum(outside, 1 - restored) * 10 ** (keep_vocals_db / 20))[:, None]

    os.makedirs(output_dir, exist_ok=True)
    speech_path = os.path.join(output_dir, "speech.wav")
    output_path = os.path.join(output_dir, "dubbed_audio.wav")
    sf.write(speech_path, np.repeat(speech[:, None], channels, axis=1), sr, subtype="PCM_16")
    final, levels = normalize(mix, sr, target_lufs, true_peak)
    sf.write(output_path, final, sr, subtype="PCM_16")
    document = {
        "audio": os.path.abspath(output_path),
        "duration": round(len(final) / sr, 3),
        "sample_rate": sr,
        "channels": channels,
        "target_lufs": target_lufs,
        "true_peak_limit": true_peak,
        "mix_lufs_before": levels["input_lufs"],
        "normalization_gain_db": levels["gain_db"],
        "output_lufs": _round(meter.integrated_loudness(final)),
        "output_true_peak": _round(20 * np.log10(max(float(_true_peaks(final).max()), 1e-9))),
        "limited_seconds": levels["limited_seconds"],
        "duck_db": duck_db,
        "keep_vocals_db": keep_vocals_db,
        "speaker_gain_db": {k: round(v, 2) for k, v in speaker_gain.items()},
        "clips": sorted(report, key=lambda r: r["id"]),
    }
    with open(os.path.join(output_dir, "mix.json"), "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved dubbed audio to {output_path} ({document['duration']:.2f}s, {sr} Hz, {channels} ch, "
          f"{document['output_lufs']} LUFS)")
    return document


def _read(path: str, duration: float, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read audio as (samples, channels), resampled to sr if given, padded or cut to exactly duration."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if sr and rate != sr:
        audio = librosa.resample(audio.T, orig_sr=rate, target_sr=sr).T
        rate = sr
    length = int(round(duration * rate))
    if len(audio) < length:
        audio = np.pad(audio, ((0, length - len(audio)), (0, 0)))
    return np.ascontiguousarray(audio[:length]), rate


def _fade(wav: np.ndarray, samples: int) -> np.ndarray:
    """Short linear fade in and out, so clips do not click."""
    samples = min(samples, len(wav) // 2)
    if samples:
        ramp = np.linspace(0, 1, samples, dtype=np.float32)
        wav = wav.copy()
        wav[:samples] *= ramp
        wav[-samples:] *= ramp[::-1]
    return wav


def _duck(envelope: np.ndarray, start: int, end: int, attack: int, release: int, gain: float):
    """Lower envelope to gain over [start, end], ramping down before start and back up after end."""
    n = len(envelope)
    envelope[max(0, start):min(n, end)] = np.minimum(envelope[max(0, start):min(n, end)], gain)
    for a, b, first, last in [(start - attack, start, 1.0, gain), (end, end + release, gain, 1.0)]:
        ramp = np.linspace(first, last, max(b - a, 1), dtype=np.float32)
        lo, hi = max(0, a), min(n, b)
        if lo < hi:
            envelope[lo:hi] = np.minimum(envelope[lo:hi], ramp[lo - a:hi - a])


def _loudness(meter: pyloudnorm.Meter, wav: np.ndarray) -> float | None:
    if len(wav) < MIN_LOUDNESS_SECONDS * meter.rate:
        return None
    value = meter.integrated_loudness(wav)
    return value if np.isfinite(value) else None


def _round(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 2)


def normalize(audio: np.ndarray, sr: int, target_lufs: float, true_peak: float,
              lookahead: float = 0.005) -> tuple[np.ndarray, dict]:
    """
    Bring audio to target_lufs with one constant gain, keeping its dynamics. Where that gain would push the
    true peak (4x oversampled) above true_peak dBTP, a look-ahead limiter lowers just those few milliseconds.
    The limiter costs a little loudness, so the gain is corrected a few times afterwards.
    """
    meter = pyloudnorm.Meter(sr)
    before = meter.integrated_loudness(audio)
    if not np.isfinite(before):
        return audio, {"input_lufs": None, "gain_db": 0.0, "limited_seconds": 0.0}

    limit = 10 ** (true_peak / 20)
    window = max(1, int(lookahead * sr))
    gain_db = target_lufs - before
    for _ in range(5):
        scaled = audio * 10 ** (gain_db / 20)
        needed = np.minimum(1.0, limit / np.maximum(_true_peaks(scaled), 1e-9))
        # min over +-window then average over window: smooth, and never above what any nearby peak needs
        smooth = uniform_filter1d(minimum_filter1d(needed, 2 * window + 1), window + 1)
        limited = scaled * np.minimum(smooth, needed)[:, None]
        error = target_lufs - meter.integrated_loudness(limited)
        if abs(error) < 0.1:
            break
        gain_db += error
    if abs(error) >= 0.5:
        print(f"⚠️ Mix is {abs(error):.1f} LU short of {target_lufs} LUFS: reaching it would need heavy limiting")
    return np.clip(limited, -1.0, 1.0), {"input_lufs": round(float(before), 2), "gain_db": round(float(gain_db), 2),
                                         "limited_seconds": round(float(np.sum(needed < 1)) / sr, 3)}


def _true_peaks(audio: np.ndarray, chunk: int = 1 << 20, margin: int = 64) -> np.ndarray:
    """Per-sample true peak across channels, estimated with 4x oversampling, in chunks to bound memory."""
    peaks = np.empty(len(audio), dtype=np.float32)
    for lo in range(0, len(audio), chunk):
        hi = min(len(audio), lo + chunk)
        a, b = max(0, lo - margin), min(len(audio), hi + margin)
        up = np.abs(resample_poly(audio[a:b], 4, 1, axis=0)).max(axis=1)
        up = up[:(b - a) * 4].reshape(-1, 4).max(axis=1)
        peaks[lo:hi] = np.maximum(up[lo - a:hi - a], np.abs(audio[lo:hi]).max(axis=1))
    return peaks
