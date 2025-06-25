import os
import srt
from pydub import AudioSegment

def create_segment_dir(segments_dir: str):
    """Create the segments directory if it doesn't exist."""
    os.makedirs(segments_dir, exist_ok=True)

def generate_audio_segments(srt_path: str, vocal_audio_path: str, segments_dir: str):
    """Generate audio segments based on subtitles."""
    # Clear existing segments directory
    print(f"[0] Clearing segments directory: {segments_dir}")
    create_segment_dir(segments_dir)


    print("[1] Parsing translated subtitles...")
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    print("[4] Loading vocals audio once...")
    vocals = AudioSegment.from_wav(vocal_audio_path)

    print("[5] Generating reference clips for each subtitle...")
    for i, sub in enumerate(subtitles):
        print(f"Subtitle {i}: '{sub.content}' from {sub.start} to {sub.end}")
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)

        clip = vocals[start_ms:end_ms]
        clip_path = os.path.join(segments_dir, f"ref_{i:04d}.wav")
        clip.export(clip_path, format="wav")
        print(f"Exported clip: {clip_path}")