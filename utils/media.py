import json
import os
import subprocess
from fractions import Fraction


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{result.stderr}")
    return result.stdout


# === Extract audio track from a video using ffmpeg ===
def extract_audio(video_path: str, audio_output_path: str, sample_rate: int | None = None, mono: bool = False):
    """Write the audio track as 16-bit PCM WAV. Keeps the source rate/channels unless overridden."""
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", "-acodec", "pcm_s16le"]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(audio_output_path)
    _run(cmd)
    print(f"✅ Extracted audio to {audio_output_path}")


# === Probe video metadata (duration, fps, audio format) using ffprobe ===
def probe_media(video_path: str, metadata_output_path: str) -> dict:
    """Save the video's duration, fps, resolution and audio format as JSON for later stages."""
    probe = json.loads(_run([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video_path
    ]))
    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)

    metadata = {
        "source": os.path.abspath(video_path),
        "duration": float(probe["format"]["duration"]),
        "video": None,
        "audio": None,
    }
    if video:
        metadata["video"] = {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": float(Fraction(video["avg_frame_rate"])) if video.get("avg_frame_rate", "0/0") != "0/0" else None,
        }
    if audio:
        metadata["audio"] = {
            "codec": audio.get("codec_name"),
            "sample_rate": int(audio["sample_rate"]),
            "channels": audio.get("channels"),
        }
    else:
        raise ValueError(f"No audio stream found in {video_path}")

    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved media metadata to {metadata_output_path} (duration: {metadata['duration']:.2f}s)")
    return metadata
