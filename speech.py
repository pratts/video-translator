from bark import SAMPLE_RATE, generate_audio, preload_models
import srt
import os
import numpy as np
from pydub import AudioSegment
from datetime import timedelta

# Environment for Bark
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# Preload models
preload_models()

# Voice prompt
voice_name = "en_speaker_6"

def bark_audio_to_segment(audio: np.ndarray) -> AudioSegment:
    # Ensure audio is in range -1.0 to 1.0
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

def convert_srt_to_audio(srt_path: str, output_path: str):
    # Load subtitles
    with open(srt_path, "r") as file:
        subtitles = list(srt.parse(file.read()))

    full_audio = AudioSegment.silent(duration=0)
    prev_end_time = 0.0  # in seconds

    for i, sub in enumerate(subtitles):
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration_ms = int((end_time - start_time) * 1000)

        print(f"\n🔊 Subtitle {i+1}: {sub.start} --> {sub.end} | \"{sub.content.strip()}\"")
        print(f"Expected duration: {duration_ms} ms")

        gap_duration_ms = int((start_time - prev_end_time) * 1000)
        if gap_duration_ms > 0:
            print(f"⏸️ Padding silence of {gap_duration_ms} ms between subtitles {i} and {i+1}")
            full_audio += AudioSegment.silent(duration=gap_duration_ms)

        # Generate audio using Bark
        audio = generate_audio(sub.content, history_prompt=voice_name)
        audio_segment = bark_audio_to_segment(audio)

        # Pad or trim to match subtitle duration
        if len(audio_segment) < duration_ms:
            print("filling silence")
            silence = AudioSegment.silent(duration=duration_ms - len(audio_segment))
            audio_segment = audio_segment.fade_out(50) + silence.fade_in(50)
        elif len(audio_segment) > duration_ms:
            print("trimming")
            audio_segment = audio_segment[:duration_ms]

        # Optional: Normalize volume
        audio_segment = audio_segment.normalize()

        # Append to final audio
        full_audio += audio_segment
        prev_end_time = end_time

    # Export final audio
    full_audio.export(output_path, format="wav")
    print(f"\n✅ Exported audio to {output_path}")

if __name__ == "__main__":
    srt_file = "./data/subtitles.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
