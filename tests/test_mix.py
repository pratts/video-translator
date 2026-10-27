import numpy as np
import pyloudnorm
import soundfile as sf

from utils.mix import _duck, _true_peaks, mix_dub, normalize


def test_duck_envelope_ramps_and_holds():
    env = np.ones(100, np.float32)
    _duck(env, 40, 60, 10, 20, 0.5)
    assert env[:30].min() == 1.0 and env[80:].min() == 1.0
    assert np.allclose(env[40:60], 0.5)
    assert np.all(np.diff(env[30:41]) <= 0) and np.all(np.diff(env[60:80]) >= 0)


def test_normalize_hits_target_and_respects_true_peak():
    sr = 44100
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((sr * 5, 2)) * 0.05).astype(np.float32)
    audio[sr:sr + 100] = 0.99                                     # a few hot peaks
    out, info = normalize(audio, sr, -14.0, -1.0)
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(out) - (-14.0)) < 0.2
    assert 20 * np.log10(_true_peaks(out).max()) <= -1.0 + 0.05
    assert info["limited_seconds"] > 0


def test_mix_keeps_length_and_leaves_background_alone_without_speech(tone, tmp_path):
    sr, duration = 44100, 6.0
    t = np.arange(int(duration * sr)) / sr
    background = np.stack([0.1 * np.sin(2 * np.pi * 110 * t)] * 2, axis=1).astype(np.float32)
    sf.write(tmp_path / "bg.wav", background[:-50], sr)           # a bit short: must be padded
    sf.write(tmp_path / "vocals.wav", background * 0.5, sr)
    report = {"clips": [{"id": 0, "speaker": "A", "start": 2.0, "audio": tone(1.0, "dub.wav", amplitude=0.2)}]}
    doc = mix_dub(report, str(tmp_path / "vocals.wav"), str(tmp_path / "bg.wav"), duration, str(tmp_path / "mix"))
    out, out_sr = sf.read(tmp_path / "mix" / "dubbed_audio.wav")
    assert out_sr == sr and out.shape == (int(duration * sr), 2)
    gain = 10 ** (doc["normalization_gain_db"] / 20)
    quiet = slice(0, int(1.5 * sr))                               # before the line and its attack ramp
    assert np.abs(out[quiet] - background[quiet] * gain).max() < 1e-3
    assert abs(doc["output_lufs"] - (-16.0)) < 0.2


def test_fallback_original_restores_the_speakers_voice_unducked(tone, tmp_path):
    sr, duration = 44100, 4.0
    t = np.arange(int(duration * sr)) / sr
    background = np.stack([0.1 * np.sin(2 * np.pi * 110 * t)] * 2, axis=1).astype(np.float32)
    vocals = np.stack([0.2 * np.sin(2 * np.pi * 330 * t)] * 2, axis=1).astype(np.float32)
    sf.write(tmp_path / "bg.wav", background, sr)
    sf.write(tmp_path / "vocals.wav", vocals, sr)
    report = {"clips": [
        {"id": 0, "speaker": "A", "start": 1.0, "end": 2.0, "status": "fallback_original", "audio": None},
        {"id": 1, "speaker": "A", "start": 2.5, "end": 2.5, "status": "fallback_silence", "audio": None}]}
    doc = mix_dub(report, str(tmp_path / "vocals.wav"), str(tmp_path / "bg.wav"), duration, str(tmp_path / "mix"),
                  keep_vocals_db=-20)
    out, _ = sf.read(tmp_path / "mix" / "dubbed_audio.wav")
    gain = 10 ** (doc["normalization_gain_db"] / 20)
    keep = 10 ** (-20 / 20)
    inside, outside = slice(int(1.1 * sr), int(1.9 * sr)), slice(int(2.6 * sr), int(3.9 * sr))
    assert np.abs(out[inside] - (background + vocals)[inside] * gain).max() < 2e-3      # original, at full level
    assert np.abs(out[outside] - (background + vocals * keep)[outside] * gain).max() < 2e-3
    assert [c["status"] for c in doc["clips"]] == ["fallback_original", "fallback_silence"]
