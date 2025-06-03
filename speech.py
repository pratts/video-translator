from bark import SAMPLE_RATE, generate_audio, preload_models
import srt
import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
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
    with open(srt_path, "r") as file:
        subtitles = list(srt.parse(file.read()))

    full_audio = AudioSegment.silent(duration=0)
    current_time_ms = 0  # Dynamic running total of audio time

    for i, sub in enumerate(subtitles):
        print(f"\n🔊 Subtitle {i+1}: {sub.start} --> {sub.end} | \"{sub.content.strip()}\"")
        
        # Generate Bark audio
        audio = generate_audio(sub.content, history_prompt=voice_name)
        audio_segment = bark_audio_to_segment(audio).normalize()
        print("generated segment: ", audio_segment.duration_seconds, " sub end:", sub.end.total_seconds(), " sub start: ", sub.start.total_seconds(), " diff: ", (sub.end - sub.start).total_seconds())
        if audio_segment.duration_seconds > (sub.end - sub.start).total_seconds():
            print("⚠️ Bark audio exceeds subtitle duration, trimming to fit")
            audio_segment = speedup(audio_segment, playback_speed=1.1, crossfade=50)

        # Get the original subtitle start time
        subtitle_start_ms = int(sub.start.total_seconds() * 1000)

        # Add silence if current time is behind original start
        if current_time_ms < subtitle_start_ms:
            gap_ms = subtitle_start_ms - current_time_ms
            print(f"⏸️ Adding {gap_ms} ms of silence before subtitle")
            full_audio += AudioSegment.silent(duration=gap_ms)
            current_time_ms += gap_ms

        # Add Bark-generated audio (without trimming)
        full_audio += audio_segment
        current_time_ms += len(audio_segment)
        audio_segment.export(f"audio_segment_{i+1}.wav", format="wav")
        # break

    # Export
    full_audio.export(output_path, format="wav")
    print(f"\n✅ Exported audio to {output_path}")


if __name__ == "__main__":
    srt_file = "./data/t_polished.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
