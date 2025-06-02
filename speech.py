from bark import SAMPLE_RATE, generate_audio
import srt
import os
import numpy as np
from pydub import AudioSegment
from datetime import timedelta

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def convert_srt_to_audio(srt_path: str, output_path: str):
    with open(srt_path, "r", encoding="utf-8") as f:
        sub = f.read()
        print("Read f: ", sub)
        subtitles = list(srt.parse(sub))

    current_time = timedelta(seconds=0)
    final_audio = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    print("Final audio initialized: ", final_audio, " current_time: ", current_time)

    for i, sub in enumerate(subtitles):
        print(f"\n 🎙️ Processing subtitle {i+1}/{len(subtitles)}: {sub.content}")

        try:
            # Generate Bark audio
            print("Generating for sub: ", sub)
            audio_array = generate_audio(sub.content)
            print("Audio array generated: ", audio_array)
            if len(audio_array) == 0:
                print(f"⚠️ Empty audio for subtitle {i+1}")
                continue

            # Convert numpy array to pydub AudioSegment
            audio_segment = AudioSegment(
                (audio_array * 32767).astype(np.int16).tobytes(),
                frame_rate=SAMPLE_RATE,
                sample_width=2,  # 16-bit PCM
                channels=1
            ).apply_gain(-2.0)  # optional: reduce loudness a bit

            print(f"Audio segment created for subtitle {i+1}, duration: {audio_segment}s")
            # Calculate how much silence we need
            silence_needed = (sub.start - current_time).total_seconds()
            print(f"Silence needed before subtitle {i+1}: {silence_needed}s current_time: {current_time}, start: {sub.start}")
            if silence_needed > 0:
                silence = AudioSegment.silent(duration=silence_needed * 1000, frame_rate=SAMPLE_RATE)
                print(f"Silence created for {silence}")
                final_audio += silence

            # Add speech
            final_audio += audio_segment

            # Update current time based on actual Bark audio duration
            delta = timedelta(seconds=audio_segment.duration_seconds)
            current_time += delta
            print(f"Delta for subtitle {i+1}: {delta} current_time updated to: {current_time}")

        except Exception as e:
            print(f"❌ Error at subtitle {i+1}: {e}")
            continue

    if final_audio.duration_seconds == 0:
        print("❌ No audio generated.")
        return

    print("Final audio duration: ", final_audio)
    final_audio.export(output_path, format="wav")
    print(f"✅ Saved to {output_path}, duration: {final_audio.duration_seconds:.2f}s")


if __name__ == "__main__":
    srt_file = "./data/translated_polished.srt"
    output_wav = "audio-output.wav"
    convert_srt_to_audio(srt_file, output_wav)
