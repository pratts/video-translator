import json
import os
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


# === Step 3a: Find who speaks when ===
def diarize(audio_path: str, diarization_output_path: str, hf_token: str,
            num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None,
            device: str | None = None) -> list[dict]:
    """
    Run pyannote speaker diarization (pyannote/speaker-diarization-community-1 via WhisperX)
    and save the speaker turns as JSON: [{"start", "end", "speaker"}].
    Requires a Hugging Face token with the model's terms accepted.
    """
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Diarization needs a Hugging Face token with access to "
                         "https://huggingface.co/pyannote/speaker-diarization-community-1")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"🔍 Loading diarization model on {device}...")
    pipeline = DiarizationPipeline(token=hf_token, device=device)

    print("🗣️ Finding speakers...")
    audio = whisperx.load_audio(audio_path)
    diarization = pipeline(audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)

    turns = [{"start": round(float(row["start"]), 3), "end": round(float(row["end"]), 3), "speaker": row["speaker"]}
             for _, row in diarization.iterrows()]

    os.makedirs(os.path.dirname(diarization_output_path), exist_ok=True)
    with open(diarization_output_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2)

    speakers = sorted({t["speaker"] for t in turns})
    print(f"✅ Saved {len(turns)} speaker turns ({len(speakers)} speakers: {', '.join(speakers)}) to {diarization_output_path}")
    return turns
