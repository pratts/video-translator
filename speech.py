from pydub import AudioSegment
from pydub.effects import speedup, normalize, compress_dynamic_range
import srt
import numpy as np
from bark import generate_audio, preload_models
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

preload_models()

def bark_audio_to_segment(audio: np.ndarray) -> AudioSegment:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return AudioSegment(audio_int16.tobytes(), frame_rate=24000, sample_width=2, channels=1)

def convert_srt_to_audio_with_timeline(srt_path: str, output_path: str):
    with open(srt_path, "r") as f:
        subtitles = list(srt.parse(f.read()))

    # Total duration of video based on last subtitle end time + some padding
    total_duration_ms = int((subtitles[-1].end.total_seconds() + 1) * 1000)

    timeline = AudioSegment.silent(duration=total_duration_ms)

    for sub in subtitles:
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        duration_ms = end_ms - start_ms

        # Generate audio with Bark
        audio = generate_audio(sub.content, history_prompt="en_speaker_6")
        audio_segment = bark_audio_to_segment(audio)

        print(f"Processing subtitle from {start_ms}ms to {end_ms}ms, duration {duration_ms} ms, length of audio segment: {len(audio_segment)} ms")

        # If audio is longer than subtitle duration, speed it up to fit
        if len(audio_segment) > duration_ms:
            speed_factor = len(audio_segment) / duration_ms
            audio_segment = speedup(audio_segment, playback_speed=speed_factor)
            audio_segment = audio_segment[:duration_ms]  # trim any tiny excess
        else:
            # If shorter, pad with silence
            silence_pad = AudioSegment.silent(duration=duration_ms - len(audio_segment))
            audio_segment = audio_segment + silence_pad

        # Normalize and compress volume
        audio_segment = compress_dynamic_range(audio_segment)
        audio_segment = normalize(audio_segment)

        # Overlay audio_segment at correct start time on timeline
        timeline = timeline.overlay(audio_segment, position=start_ms)

    timeline.export(output_path, format="wav")
    print(f"Exported final audio with timeline to {output_path}")
