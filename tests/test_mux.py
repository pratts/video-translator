import json
import subprocess

import numpy as np
import soundfile as sf

from utils.mux import mux_video


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name:"
                          "stream_tags=language:stream_disposition=default", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_mux_tracks_languages_and_length(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(video)], check=True)
    sr = 44100
    sf.write(tmp_path / "dub.wav", np.zeros((3 * sr, 2), np.float32), sr)
    doc = {"source_language": "ja", "target_language": "en", "segments": [
        {"start": 0.5, "end": 1.5, "translation": "Hello.", "source_text": "こんにちは。"},
        {"start": 2.0, "end": 2.5, "translation": "", "source_text": "はい"}]}
    out = mux_video(str(video), str(tmp_path / "dub.wav"), doc, str(tmp_path / "out"))
    assert out.endswith("clip.en.mp4")
    info = probe(out)
    streams = [(s["codec_type"], s.get("tags", {}).get("language"), s["disposition"]["default"]) for s in info["streams"]]
    assert streams == [("video", "und", 1), ("audio", "eng", 1), ("audio", "jpn", 0),
                       ("subtitle", "eng", 1), ("subtitle", "jpn", 0)]
    assert abs(float(info["format"]["duration"]) - 3.0) < 0.1
    english = (tmp_path / "out" / "clip.en.subtitles" / "en.srt").read_text()
    assert "Hello." in english and english.count("-->") == 1     # empty lines get no cue
