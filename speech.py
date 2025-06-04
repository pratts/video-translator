from bark import SAMPLE_RATE, generate_audio, preload_models
import srt
import os
import numpy as np
from pydub import AudioSegment
from datetime import timedelta

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
preload_models()

voice_name = "en_speaker_6"

def bark_audio_to_segment(audio: np.ndarray) -> AudioSegment:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )

def change_speed(sound: AudioSegment, speed: float) -> AudioSegment:
    """Change playback speed without affecting pitch."""
    new_frame_rate = int(sound.frame_rate * speed)
    sped_up = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
    return sped_up.set_frame_rate(sound.frame_rate)

def convert_srt_to_audio(srt_path: str, output_path: str):
    with open(srt_path, "r") as file:
        subtitles = list(srt.parse(file.read()))

    full_audio = AudioSegment.silent(duration=0)

    for i, sub in enumerate(subtitles):
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration_ms = int((end_time - start_time) * 1000)

        print(f"\n🔊 Subtitle {i+1}: {sub.start} --> {sub.end} | \"{sub.content.strip()}\"")
        print(f"Expected duration: {duration_ms} ms")

        audio = generate_audio(sub.content, history_prompt=voice_name)
        audio_segment = bark_audio_to_segment(audio)

        original_length = len(audio_segment)

        print("Original length: ", original_length, " ms, Duration: ", duration_ms, " ms", " start: ", start_time, " end: ", end_time)
        # Match audio to subtitle duration
        if original_length < duration_ms:
            print("📭 Padding with silence")
            silence = AudioSegment.silent(duration=duration_ms - original_length)
            audio_segment = audio_segment.fade_out(50) + silence.fade_in(50)
        elif original_length > duration_ms:
            speed = duration_ms / original_length
            print("speed: ", speed)
            if 0.9 <= speed <= 1.1:
                print(f"⚡ Speeding up to fit: {speed:.2f}x")
                audio_segment = change_speed(audio_segment, speed)
            else:
                print("⛔ Audio too long to safely speed up, trimming")
                # audio_segment = audio_segment[:duration_ms]

        audio_segment = audio_segment.normalize()
        full_audio += audio_segment

    full_audio.export(output_path, format="wav")
    print(f"\n✅ Exported final audio to {output_path}")

if __name__ == "__main__":
    srt_file = "./data/t_polished.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
