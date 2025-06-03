from bark import generate_audio
from pydub import AudioSegment
import srt
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

final_audio = AudioSegment.silent(duration=0)

with open("translated.srt", "r", encoding="utf-8") as f:
    subs = list(srt.parse(f.read()))

print(f"Total subtitles: {len(subs)}")
for sub in subs:
    duration = (sub.end - sub.start).total_seconds() * 1000
    print(f"Generating: {sub.content}")
    audio_array = generate_audio(sub.content, history_prompt="v2/en_speaker_1")
    clip = AudioSegment(audio_array.tobytes(), frame_rate=24000, sample_width=2, channels=1)

    final_audio += clip
    # Add silence between segments if needed (calculate from subtitle gaps)

final_audio.export("final_audio.wav", format="wav")
