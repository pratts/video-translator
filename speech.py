from bark import SAMPLE_RATE, generate_audio
import srt
import os
import numpy as np
from scipy.io.wavfile import write as write_wav

from bark import preload_models
from bark.generation import preload_models

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# Preload necessary Bark models
preload_models()

# Set voice and text prompt
voice_name = "en_speaker_6"
silence_threshold = 2000  # ms

def convert_srt_to_audio(srt_path: str, output_path: str):
    # Load SRT
    with open(srt_path, "r") as file:
        subtitles = list(srt.parse(file.read()))

    # Initial state
    audio_arr = []

    print("\n\n")
    # Process each subtitle
    for i, sub in enumerate(subtitles):
        print(f"\nSubtitle {i+1}: {sub.start} --> {sub.end} | {sub.content.strip()}")
        
        audio = generate_audio(sub.content,history_prompt=voice_name)
        
        print("Generated audio :", audio)
        write_wav(f"audio{i+1}.wav", SAMPLE_RATE, audio)
        audio_arr.append(audio)

    # Export final audio
    output = np.concatenate(audio_arr)
    write_wav(output_path, SAMPLE_RATE, output)
    print(f"\n✅ Exported audio to {output_path}")


if __name__ == "__main__":
    srt_file = "./data/translated_polished.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
