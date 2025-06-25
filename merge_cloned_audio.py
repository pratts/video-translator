from pydub import AudioSegment
import srt
import os

def merge_openvoice_speech_with_subtitles(srt_path: str, open_voice_path: str, output_path: str):
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    # Total timeline duration = last subtitle's end + 1 second padding
    total_duration_ms = int((subtitles[-1].end.total_seconds() + 1) * 1000)
    timeline = AudioSegment.silent(duration=total_duration_ms)

    for i, sub in enumerate(subtitles):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        expected_duration = end_ms - start_ms

        audio_path = os.path.join(open_voice_path, f"converted_{i:04d}.wav")
        print(f"[🔊] Processing subtitle {i}: {sub.content} (start: {sub.start}, end: {sub.end}), path: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"[⚠️] Missing audio for subtitle {i}: {audio_path}")
            continue

        audio_segment = AudioSegment.from_wav(audio_path)

        # Pad with silence if shorter than expected (helps with alignment)
        if len(audio_segment) < expected_duration:
            pad_ms = expected_duration - len(audio_segment)
            audio_segment += AudioSegment.silent(duration=pad_ms)

        timeline = timeline.overlay(audio_segment, position=start_ms)

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
    open_voice_path = "./data/processed"
    polished_subtitle_file = './data/translated_polished.srt'
    output_audio_file = "./data/merged_openvoice_audio.wav"
    bgm_file = './data/htdemucs/source-video/no_vocals.wav'
    merged_audio_file = './data/merged_audio.wav'
    clear_files([output_audio_file, merged_audio_file])

    merge_openvoice_speech_with_subtitles(polished_subtitle_file, open_voice_path, output_audio_file)

    print("Step 5: Merging cloned audio with background music...")
    merge_voice_with_bgm(output_audio_file, bgm_file, merged_audio_file)