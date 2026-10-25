import json
import os
import srt
import torch
import whisperx
from datetime import timedelta

ASR_SAMPLE_RATE = 16000
MLX_MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}


# === Step 2: Transcribe vocals with word-level timestamps ===
def transcribe(audio_path: str, transcript_output_path: str, srt_output_path: str,
               backend: str = "whisperx", model_name: str = "large-v3-turbo",
               language: str | None = None, device: str | None = None) -> dict:
    """
    Transcribe speech and align every word to the audio.
    - backend "whisperx": faster-whisper + VAD (CUDA or CPU)
    - backend "mlx": mlx-whisper (fast on Apple Silicon)
    Both are followed by WhisperX wav2vec2 forced alignment for accurate word timestamps.
    Saves the full result as JSON and a segment-level SRT for display/translation.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio = whisperx.load_audio(audio_path, sr=ASR_SAMPLE_RATE)

    print(f"🎙️ Transcribing with {backend} ({model_name}) on {device if backend == 'whisperx' else 'mlx'}...")
    if backend == "whisperx":
        result = _transcribe_whisperx(audio, model_name, language, device)
    elif backend == "mlx":
        result = _transcribe_mlx(audio, model_name, language)
    else:
        raise ValueError(f"Unknown ASR backend: {backend}. Use 'whisperx' or 'mlx'.")

    detected_language = result["language"]
    print(f"🌐 Language: {detected_language}")

    print("📐 Aligning words to audio...")
    align_model, align_metadata = whisperx.load_align_model(language_code=detected_language, device=device)
    aligned = whisperx.align(result["segments"], align_model, align_metadata, audio, device,
                             return_char_alignments=False)

    transcript = {
        "language": detected_language,
        "backend": backend,
        "model": model_name,
        "audio": os.path.abspath(audio_path),
        "segments": [_clean_segment(i, s) for i, s in enumerate(aligned["segments"])],
    }

    os.makedirs(os.path.dirname(transcript_output_path), exist_ok=True)
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved transcript to {transcript_output_path} "
          f"({len(transcript['segments'])} segments, {sum(len(s['words']) for s in transcript['segments'])} words)")

    write_srt(transcript, srt_output_path)
    return transcript


def _transcribe_whisperx(audio, model_name: str, language: str | None, device: str) -> dict:
    # ctranslate2 has no Apple GPU support, so use int8 on CPU and float16 on CUDA
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    return model.transcribe(audio, batch_size=16 if device == "cuda" else 4, language=language)


def _transcribe_mlx(audio, model_name: str, language: str | None) -> dict:
    import mlx_whisper  # only available on Apple Silicon

    repo = MLX_MODELS.get(model_name, model_name)
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=language,
                                    condition_on_previous_text=False)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result["segments"] if s["text"].strip()]
    return {"segments": segments, "language": result["language"]}


def _clean_segment(index: int, segment: dict) -> dict:
    """Keep only what later stages need, and fill words that alignment could not time (e.g. numbers)."""
    words = [{"word": w["word"], "start": w.get("start"), "end": w.get("end"), "score": w.get("score")}
             for w in segment.get("words", [])]

    for i, word in enumerate(words):
        if word["start"] is None:
            word["start"] = words[i - 1]["end"] if i > 0 else segment["start"]
        if word["end"] is None:
            next_start = next((w["start"] for w in words[i + 1:] if w["start"] is not None), None)
            word["end"] = next_start if next_start is not None else segment["end"]

    return {
        "id": index,
        "start": round(segment["start"], 3),
        "end": round(segment["end"], 3),
        "text": segment["text"].strip(),
        "words": words,
    }


def write_srt(transcript: dict, srt_output_path: str):
    subtitles = [
        srt.Subtitle(index=s["id"] + 1,
                     start=timedelta(seconds=s["start"]),
                     end=timedelta(seconds=s["end"]),
                     content=s["text"])
        for s in transcript["segments"]
    ]
    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    print(f"✅ Saved subtitles to {srt_output_path}")
