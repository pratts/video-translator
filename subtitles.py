import whisper
import srt
import os
from datetime import timedelta
from moviepy import VideoFileClip

# === Step 1: Extract Audio from Original Video ===

def extract_audio(video_path: str, audio_output_path: str):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    print(f"✅ Extracted audio to {audio_output_path}")


# === Step 2: Transcribe Audio Using Whisper ===

def transcribe_to_srt(audio_path: str, srt_output_path: str, model_size="medium"):
    print("🔍 Loading Whisper model...")
    model = whisper.load_model(model_size)

    print("🎙️ Transcribing...")
    result = model.transcribe(audio_path)

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


# === Run the pipeline ===

if __name__ == "__main__":
    video_file = "./data/source-video.mp4"
    extracted_audio = "./data/source-audio.wav"
    subtitle_file = "./data/subtitles.srt"

    extract_audio(video_file, extracted_audio)
    transcribe_to_srt(extracted_audio, subtitle_file, model_size="turbo")  # or "large"
