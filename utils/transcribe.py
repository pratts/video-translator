import whisper
import srt
import os
from datetime import timedelta

# === Step 2: Transcribe Audio Using Whisper ===
def transcribe_to_srt(audio_path: str, srt_output_path: str, model_size="medium"):
    print("🔍 Loading Whisper model...")
    model = whisper.load_model(model_size)

    print("🎙️ Transcribing...")
    result = model.transcribe(audio_path, word_timestamps=True)

    subtitles = []
    for i, segment in enumerate(result["segments"]):
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        text = segment["text"].strip()

        subtitles.append(
            srt.Subtitle(index=i + 1, start=start, end=end, content=text)
        )

    with open(srt_output_path, "w") as f:
        f.write(srt.compose(subtitles))
    
    print(f"✅ Saved subtitles to {srt_output_path}")
