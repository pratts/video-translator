from pathlib import Path
import demucs.api


# === Separate speech from background using Demucs ===
def separate_vocals(audio_path: str, vocals_output_path: str, background_output_path: str,
                    model: str = "htdemucs_ft", device: str | None = None):
    """
    Split the audio into vocals and background (everything that is not vocals).
    Uses the maintained Demucs (https://github.com/adefossez/demucs) Python API.
    """
    print(f"🔍 Loading Demucs model '{model}'...")
    options = {"model": model, "progress": True}
    if device:
        options["device"] = device
    separator = demucs.api.Separator(**options)

    print("🎚️ Separating vocals from background...")
    _, stems = separator.separate_audio_file(Path(audio_path))

    vocals = stems["vocals"]
    background = sum(wav for name, wav in stems.items() if name != "vocals")

    for wav, output_path in [(vocals, vocals_output_path), (background, background_output_path)]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # clamp instead of per-stem rescaling, so vocals/background keep their original relative levels
        demucs.api.save_audio(wav, output_path, samplerate=separator.samplerate, clip="clamp")
        print(f"✅ Saved {output_path}")
