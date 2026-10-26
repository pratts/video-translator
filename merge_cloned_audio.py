from pydub import AudioSegment
import json
import os

def merge_fitted_clips(timing_report_path: str, output_path: str, duration: float):
    """Place every fitted clip at its start time on a timeline as long as the video (Step 6 guarantees no overlaps)."""
    with open(timing_report_path, encoding="utf-8") as f:
        clips = json.load(f)["clips"]

    timeline = AudioSegment.silent(duration=int(duration * 1000))
    for clip in clips:
        print(f"[🔊] Placing segment {clip['id']} at {clip['start']:.2f}s: {clip['text']}")
        timeline = timeline.overlay(AudioSegment.from_wav(clip["audio"]), position=int(clip["start"] * 1000))

    timeline.export(output_path, format="wav")
    print(f"[✅] Exported final audio to: {output_path}")

def clear_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleared file: {file}")
        else:
            print(f"File not found, skipping: {file}")

def merge_voice_with_bgm(source_audio_path, bgm_path, output_path):
    # Load the source audio and background music
    source_audio = AudioSegment.from_wav(source_audio_path)
    bgm = AudioSegment.from_wav(bgm_path)

    # Match background music duration if needed
    bgm = bgm[:len(source_audio)] if len(bgm) > len(source_audio) else bgm + AudioSegment.silent(duration=len(source_audio)-len(bgm))

    # Mix with proper gain to avoid overpowering
    mixed = source_audio.overlay(bgm - 6)
    mixed.export(output_path, format="wav")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    translator = os.getenv("TRANSLATOR", "google").lower()
    timing_report_file = f"./data/translations/{translator}/fitted/timing_report.json"
    with open("./data/metadata.json") as f:
        video_duration = json.load(f)["duration"]
    output_audio_file = "./data/merged_openvoice_audio.wav"
    bgm_file = './data/separation/background.wav'
    merged_audio_file = './data/merged_audio.wav'
    clear_files([output_audio_file, merged_audio_file])

    merge_fitted_clips(timing_report_file, output_audio_file, video_duration)

    print("Step 5: Merging cloned audio with background music...")
    merge_voice_with_bgm(output_audio_file, bgm_file, merged_audio_file)