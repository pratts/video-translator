import srt
import os
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def convert_srt_to_audio(srt_path: str, output_path: str):
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    audio_segments = []

    print(f"Generating audio for {len(subtitles)} segments...")

    for i, sub in enumerate(subtitles):
        print(f"Processing segment {i+1}/{len(subtitles)}...")
        audio_array = generate_audio(sub.content)
        audio_segments.append((sub.start.total_seconds(), audio_array))

    # Combine into a full audio stream (simple naive concatenation)
    full_audio = b''.join(audio[1].tobytes() for audio in audio_segments)

    # Save to WAV
    with open(output_path, "wb") as f:
        f.write(full_audio)

    print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    srt_file = "./data/translated_polished.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
