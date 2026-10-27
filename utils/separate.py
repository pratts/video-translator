import math
from pathlib import Path

import demucs.api
from tqdm import tqdm


# === Separate speech from background using Demucs ===
def separate_vocals(audio_path: str, vocals_output_path: str, background_output_path: str,
                    model: str = "htdemucs_ft", device: str | None = None):
    """
    Split the audio into vocals and background (everything that is not vocals).
    Uses the maintained Demucs (https://github.com/adefossez/demucs) Python API.
    """
    print(f"🔍 Loading Demucs model '{model}'...")
    progress = _Progress()
    options = {"model": model, "callback": progress}
    if device:
        options["device"] = device
    separator = demucs.api.Separator(**options)
    progress.separator = separator

    print("🎚️ Separating vocals from background...")
    try:
        _, stems = separator.separate_audio_file(Path(audio_path))
    finally:
        progress.close()

    vocals = stems["vocals"]
    background = sum(wav for name, wav in stems.items() if name != "vocals")

    for wav, output_path in [(vocals, vocals_output_path), (background, background_output_path)]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # clamp instead of per-stem rescaling, so vocals/background keep their original relative levels
        demucs.api.save_audio(wav, output_path, samplerate=separator.samplerate, clip="clamp")
        print(f"✅ Saved {output_path}")


class _Progress:
    """
    One progress bar for the whole separation. Demucs' own bar restarts for every model in a bag
    (htdemucs_ft is 4 models), so it looks like the same step running 4 times.
    """

    def __init__(self):
        self.separator, self.bar = None, None

    def __call__(self, info: dict):
        if self.bar is None:
            self.bar = tqdm(total=self._chunks(info), desc="Separating", unit="chunk", leave=False)
        if info["state"] == "end":
            self.bar.update(1)

    def _chunks(self, info: dict) -> int | None:
        """Chunks Demucs will process: per model and shift, the audio split into overlapping segments."""
        try:
            model = self.separator.model
            models = getattr(model, "models", [model])
            segment = float(self.separator._segment or models[0].segment)
            stride = int((1 - self.separator._overlap) * int(model.samplerate * segment))
            return len(models) * max(1, self.separator._shifts) * math.ceil(info["audio_length"] / stride)
        except (AttributeError, TypeError, ZeroDivisionError):
            return None  # unknown total: the bar just counts

    def close(self):
        if self.bar is not None:
            self.bar.close()
