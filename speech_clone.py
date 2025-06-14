import srt
from bark import generate_audio, SAMPLE_RATE
from pydub import AudioSegment
import numpy as np
import torch
import os
import pyrubberband as pyrb
import soundfile as sf
import io

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def get_audio_segment(audio_path: str, start: float, end: float):
    audio = AudioSegment.from_file(audio_path)
    segment = audio[int(start * 1000):int(end * 1000)]
    return segment.normalize()

def generate_bark_audio(content: str) -> np.ndarray:
    return generate_audio(content, history_prompt="v2/en_speaker_6")

def stretch_pydub_segment(segment: AudioSegment, target_duration_sec: float) -> AudioSegment:
    print(f"Stretching audio segment from {len(segment)/1000.0:.2f}s to {target_duration_sec:.2f}s")
    samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))

    sr = segment.frame_rate
    rate = len(segment) / 1000.0 / target_duration_sec
    stretched = pyrb.time_stretch(samples, sr, rate)

    buf = io.BytesIO()
    sf.write(buf, stretched, sr, format='WAV')
    buf.seek(0)
    return AudioSegment.from_file(buf, format='wav')

def merge_adjacent_subtitles(subtitles, max_gap=0.5, max_duration=8.0):
    merged = []
    group = []
    group_start = None

    for sub in subtitles:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()

        if not group:
            group = [sub]
            group_start = start
            continue

        prev_end = group[-1].end.total_seconds()
        group_end = end
        duration = group_end - group_start

        if (start - prev_end) <= max_gap and duration <= max_duration:
            group.append(sub)
        else:
            merged.append(group)
            group = [sub]
            group_start = start

    if group:
        merged.append(group)

    return merged

def clone_speech_with_subtitles(audio_path: str, srt_path: str, output_path: str):
    with open(srt_path, "r") as f:
        raw_subs = list(srt.parse(f.read()))
    merged_subs = merge_adjacent_subtitles(raw_subs)

    final_audio = AudioSegment.silent(duration=0)
    last_end_ms = 0

    for group in merged_subs:
        start = group[0].start.total_seconds()
        end = group[-1].end.total_seconds()
        content = " ".join([s.content.strip() for s in group])
        print(f"\nProcessing merged sub: {start:.2f}s - {end:.2f}s: {content}")

        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        if start_ms > last_end_ms:
            final_audio += AudioSegment.silent(duration=start_ms - last_end_ms)

        ref_audio = get_audio_segment(audio_path, start, end)
        audio_array = generate_bark_audio(content)

        # Normalize, convert to 16-bit PCM AudioSegment
        audio_segment = AudioSegment(
            (np.clip(audio_array, -1, 1) * 32767).astype(np.int16).tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )

        original_duration = end - start
        if len(audio_segment) / 1000.0 > original_duration:
            audio_segment = stretch_pydub_segment(audio_segment, original_duration)
        elif len(audio_segment) / 1000.0 < original_duration:
            pad_duration = int(original_duration * 1000) - len(audio_segment)
            audio_segment += AudioSegment.silent(duration=pad_duration)

        final_audio += audio_segment
        last_end_ms = start_ms + len(audio_segment)

    final_audio.export(output_path, format="wav")
    print(f"\n✅ Exported final audio to {output_path}")
