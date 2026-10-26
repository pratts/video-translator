import json
import os
import subprocess
from datetime import timedelta

import srt

# ISO 639-1 (as used in the pipeline) to ISO 639-2, which MP4/MKV language tags use
ISO_639_2 = {
    "ar": "ara", "bn": "ben", "de": "deu", "en": "eng", "es": "spa", "fr": "fra", "hi": "hin", "id": "ind",
    "it": "ita", "ja": "jpn", "ko": "kor", "nl": "nld", "pl": "pol", "pt": "por", "ru": "rus", "ta": "tam",
    "te": "tel", "th": "tha", "tr": "tur", "uk": "ukr", "ur": "urd", "vi": "vie", "yue": "yue", "zh": "zho",
}


# === Step 8: Put the dubbed audio and subtitles into the final video ===
def mux_video(video_path: str, dubbed_audio_path: str, translation_doc: dict, output_dir: str,
              keep_original_audio: bool = True, original_subtitles: bool = True,
              audio_bitrate: str = "192k") -> str:
    """
    Write output/<name>.<target language>.<ext> (MP4, or MKV for MKV input) with:
    - the source video stream copied without re-encoding
    - the dubbed audio (AAC, default track), and optionally the source's own audio, copied as a second track
    - translated subtitles, and optionally original-language subtitles, built from the dubbing segments
    Every track is tagged with its language. Returns the output path.
    """
    source, target = translation_doc["source_language"], translation_doc["target_language"]
    name, ext = os.path.splitext(os.path.basename(video_path))
    mkv = ext.lower() == ".mkv"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.{target}{'.mkv' if mkv else '.mp4'}")

    subtitle_dir = os.path.join(output_dir, f"{name}.{target}.subtitles")
    tracks = [(target, write_subtitles(translation_doc, "translation", os.path.join(subtitle_dir, f"{target}.srt")))]
    if original_subtitles:
        tracks.append((source, write_subtitles(translation_doc, "source_text",
                                               os.path.join(subtitle_dir, f"{source}.srt"))))

    inputs = [video_path, dubbed_audio_path] + [path for _, path in tracks]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        cmd += ["-i", path]

    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    audio = [(target, f"{_language_name(target)} (dubbed)")]
    if keep_original_audio:
        cmd += ["-map", "0:a:0?"]
        audio.append((source, f"{_language_name(source)} (original)"))
    for i in range(len(tracks)):
        cmd += ["-map", f"{2 + i}:s:0"]

    # the original track is copied as is; only the dub is encoded
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a:0", audio_bitrate, "-c:a:1", "copy",
            "-c:s", "srt" if mkv else "mov_text", "-map_metadata", "0", "-map_chapters", "0"]
    for i, (language, title) in enumerate(audio):
        cmd += [f"-metadata:s:a:{i}", f"language={_iso639_2(language)}", f"-metadata:s:a:{i}", f"title={title}",
                f"-disposition:a:{i}", "default" if i == 0 else "0"]
    for i, (language, _) in enumerate(tracks):
        kind = "translated" if i == 0 else "original"
        cmd += [f"-metadata:s:s:{i}", f"language={_iso639_2(language)}",
                f"-metadata:s:s:{i}", f"title={_language_name(language)} ({kind})", f"-disposition:s:{i}", "0"]
    if not mkv:
        cmd += ["-movflags", "+faststart"]
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr}")

    source_duration, output_duration = _duration(video_path), _duration(output_path)
    if abs(source_duration - output_duration) > 0.1:
        print(f"⚠️ Output is {output_duration:.2f}s but the source video is {source_duration:.2f}s")
    print(f"✅ Saved {output_path} ({output_duration:.2f}s; audio: {', '.join(t for _, t in audio)}; "
          f"subtitles: {', '.join(_language_name(l) for l, _ in tracks)})")
    return output_path


def write_subtitles(translation_doc: dict, field: str, path: str) -> str:
    """One subtitle per dubbing segment, with the segment's timing (no speaker labels)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subtitles = [srt.Subtitle(index=i + 1, start=timedelta(seconds=s["start"]), end=timedelta(seconds=s["end"]),
                              content=s[field]) for i, s in enumerate(translation_doc["segments"]) if s[field].strip()]
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    return path


def _iso639_2(language: str) -> str:
    return ISO_639_2.get(language, language)


def _language_name(language: str) -> str:
    names = {"en": "English", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
             "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ko": "Korean", "ru": "Russian"}
    return names.get(language, language)


def _duration(path: str) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                            capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])
