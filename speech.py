from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
import srt
import numpy as np
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def convert_srt_to_audio(srt_path: str, output_path: str):
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    full_audio = []

    for i, sub in enumerate(subtitles):
        print(f"Processing subtitle {i+1}/{len(subtitles)}: {sub.content}")

        try:
            audio_array = generate_audio(sub.content)

            if len(audio_array) == 0:
                print(f"⚠️ Warning: Empty audio for subtitle {i+1}")
                continue

            full_audio.append(audio_array)
        except Exception as e:
            print(f"❌ Error processing subtitle {i+1}: {e}")
            continue

    if not full_audio:
        print("❌ No audio segments generated. Exiting.")
        return

    combined_audio = np.concatenate(full_audio)
    write_wav(output_path, SAMPLE_RATE, combined_audio)

    print(f"✅ Audio saved to {output_path}, length: {len(combined_audio)/SAMPLE_RATE:.2f} sec")


if __name__ == "__main__":
    srt_file = "./data/translated_polished.srt"
    output_wav = "audio-output_prev.wav"
    convert_srt_to_audio(srt_file, output_wav)
