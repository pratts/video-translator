"""Full pipeline on a 12 s cut of the sample video with real models. Opt-in: RUN_E2E=1 pytest -m e2e
Needs the models (downloaded on first run), HF_TOKEN for diarization, and uses the local nllb translator."""
import json
import os
import subprocess

import pytest

import convert

pytestmark = [pytest.mark.e2e,
              pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="set RUN_E2E=1 to run the full pipeline")]
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "source-video.mp4")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def test_dubs_a_short_clip_and_resumes(tmp_path, capsys):
    clip = tmp_path / "sample.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", SAMPLE, "-t", "12", "-c", "copy", str(clip)], check=True)
    args = [str(clip), "--tgt", "en", "--translator", "nllb", "--work-dir", str(tmp_path / "work"),
            "--output-dir", str(tmp_path / "out")]
    convert.main(args)
    output = tmp_path / "out" / "sample.en.mp4"
    assert output.exists() and abs(duration(str(output)) - duration(str(clip))) < 0.1

    capsys.readouterr()
    convert.main(args)                                   # second run: everything is up to date
    assert capsys.readouterr().out.count("up to date, skipping") == len(convert.STAGES)
