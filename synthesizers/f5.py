import os
from functools import cache

import numpy as np

SAMPLE_RATE = 24000
# generates speech of a requested length (target_duration)
SUPPORTS_DURATION = True
# Lines this short come out garbled when squeezed into a tiny slot, so give them a little more room
MIN_DURATION = 0.8


@cache
def _model():
    import torch
    from f5_tts.api import F5TTS

    device = os.getenv("TTS_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    name = os.getenv("F5_MODEL") or "F5TTS_v1_Base"
    print(f"🔍 Loading F5-TTS ({name}) on {device}...")
    return F5TTS(model=name, device=device)


def synthesize(text: str, reference: dict, language: str, target_duration: float | None = None,
               seed: int | None = 0) -> tuple[np.ndarray, int]:
    """
    Clone the reference voice saying text. reference = {"audio": wav path, "text": its transcript, "language"}.
    With target_duration, F5-TTS generates a clip of exactly that length (it sets the total
    mel length = reference + target). The clip includes some silence at its edges, so the speech itself is
    shorter; utils/synthesize.py measures it and asks for more room when needed.
    F5TTS_v1_Base speaks English and Chinese.
    """
    model = _model()
    fix_duration = None
    if target_duration:
        fix_duration = _reference_seconds(reference["audio"]) + max(target_duration, MIN_DURATION)
    ref_text = _reference_text(reference["text"], reference.get("language"))
    wav, sr, _ = model.infer(ref_file=reference["audio"], ref_text=ref_text, gen_text=text,
                             fix_duration=fix_duration, seed=seed, show_info=lambda *a: None, progress=None)
    return np.asarray(wav, dtype=np.float32), sr


@cache
def _reference_seconds(path: str) -> float:
    # F5-TTS trims the reference's silent edges and adds 50 ms before measuring it, so do the same
    from pydub import AudioSegment
    from f5_tts.infer.utils_infer import remove_silence_edges
    return len(remove_silence_edges(AudioSegment.from_file(path)) + AudioSegment.silent(duration=50)) / 1000


@cache
def _reference_text(text: str, language: str | None) -> str:
    """
    F5-TTS reads Latin letters and Chinese characters only: kana are unknown to it and kanji are read as
    Chinese. A transcript that does not match the reference audio makes it repeat reference sounds at the
    start of the generated line, so Japanese transcripts are romanized first.
    """
    if language != "ja":
        return text
    import cutlet
    katsu = cutlet.Cutlet()
    katsu.use_foreign_spelling = False
    return katsu.romaji(text)
