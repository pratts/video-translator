import argparse
import json
import os
import shutil
import sys
from dotenv import load_dotenv

# Load .env before importing the pipeline modules, so secrets and settings are available when they load
load_dotenv()

from utils import output
from utils.config import load_config
from utils.pipeline import Pipeline, Stage, file_id
from utils.mux import output_path_for

# Each stage imports its own modules when it runs, so --help, --clean and skipped stages
# do not load the heavy model libraries


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Run:
    """Paths and settings for dubbing one video. Everything for the video lives in <work_dir>/<video name>/."""

    def __init__(self, video: str, config: dict, work_root: str = "work", output_dir: str = "output"):
        self.video = os.path.abspath(video)
        self.config = config
        self.name = os.path.splitext(os.path.basename(video))[0]
        self.work_dir = os.path.abspath(os.path.join(work_root, self.name))
        self.output_dir = os.path.abspath(output_dir)
        w = lambda *p: os.path.join(self.work_dir, *p)

        self.metadata = w("metadata.json")
        self.source_audio = w("source-audio.wav")
        self.vocals = w("separation", "vocals.wav")
        self.background = w("separation", "background.wav")
        self.transcript = w("transcript.json")
        self.transcript_srt = w("subtitles.srt")
        self.diarization = w("diarization.json")
        self.segments = w("segments.json")
        self.segments_srt = w("segments.srt")
        self.voices_dir = w("voices")
        self.references = w("voices", "references.json")
        # each translator has its own folder, and every later stage reads from the selected translator's folder
        self.translation_dir = w("translations", config["TRANSLATOR"])
        self.translation = os.path.join(self.translation_dir, "translation.json")
        self.raw_srt = os.path.join(self.translation_dir, "translated_raw.srt")
        self.polished_srt = os.path.join(self.translation_dir, "translated_polished.srt")
        self.clips_dir = os.path.join(self.translation_dir, "clips")
        self.fitted_dir = os.path.join(self.translation_dir, "fitted")
        self.mix_dir = os.path.join(self.translation_dir, "mix")

    def duration(self) -> float:
        return read_json(self.metadata)["duration"]

    def tts_options(self) -> dict:
        c = self.config
        return dict(backend=c["TTS_BACKEND"], verify=c["TTS_VERIFY"], attempts=c["TTS_ATTEMPTS"],
                    asr_model=c["WHISPER_MODEL"], max_speed=c["FIT_MAX_SPEED"])

    def clips_doc(self) -> str:
        return os.path.join(self.clips_dir, "clips.json")


# === Stages ===
def extract(run: Run):
    from utils.media import extract_audio, probe_media
    probe_media(run.video, run.metadata)
    extract_audio(run.video, run.source_audio)


def separate(run: Run):
    from utils.separate import separate_vocals
    c = run.config
    separate_vocals(run.source_audio, run.vocals, run.background, model=c["DEMUCS_MODEL"],
                    device=c["DEMUCS_DEVICE"] or None, shifts=c["DEMUCS_SHIFTS"], seed=c["SEED"])


def transcribe_stage(run: Run):
    from utils.transcribe import transcribe
    c = run.config
    transcribe(run.vocals, run.transcript, run.transcript_srt, backend=c["ASR_BACKEND"], model_name=c["WHISPER_MODEL"],
               language=c["SOURCE_LANGUAGE"] or None, device=c["ASR_DEVICE"] or None, glossary=c["GLOSSARY"])


def diarize_stage(run: Run):
    from utils.diarize import diarize
    c = run.config
    diarize(run.vocals, run.diarization, os.getenv("HF_TOKEN"), num_speakers=c["NUM_SPEAKERS"],
            min_speakers=c["MIN_SPEAKERS"], max_speakers=c["MAX_SPEAKERS"], device=c["DIARIZATION_DEVICE"] or None)


def segment(run: Run):
    from utils.segment import build_segments, save_segments
    segments = build_segments(read_json(run.transcript), read_json(run.diarization), run.duration(),
                              min_duration=run.config["SEGMENT_MIN_DURATION"],
                              merge_gap=run.config["SEGMENT_MERGE_GAP"])
    save_segments(segments, run.segments, run.segments_srt)


def voices(run: Run):
    from utils.voices import build_voice_references
    build_voice_references(read_json(run.segments), read_json(run.diarization), run.vocals, run.background,
                           run.voices_dir)


def translate(run: Run):
    from utils.translate import translate_segments
    c = run.config
    translate_segments(read_json(run.segments), run.translation, run.raw_srt, run.polished_srt,
                       translator=c["TRANSLATOR"], target_language=c["TARGET_LANGUAGE"],
                       glossary=c["GLOSSARY"], context=c["VIDEO_CONTEXT"] or None)


def tts(run: Run):
    from utils.synthesize import synthesize_segments
    synthesize_segments(read_json(run.translation), read_json(run.references), run.clips_dir, **run.tts_options(),
                        on_mismatch=run.config["TTS_ON_MISMATCH"], translation_path=run.translation)


def fit(run: Run):
    from utils.fit import fit_clips
    from utils.synthesize import make_regenerator
    c = run.config
    regenerate = make_regenerator(read_json(run.translation), run.translation, read_json(run.references),
                                  **run.tts_options())
    fit_clips(read_json(run.clips_doc()), run.fitted_dir, video_duration=run.duration(),
              min_speed=c["FIT_MIN_SPEED"], max_speed=c["FIT_MAX_SPEED"], hard_max_speed=c["FIT_HARD_MAX_SPEED"],
              max_borrow=c["FIT_MAX_BORROW"], max_retries=c["FIT_MAX_RETRANSLATIONS"], regenerate=regenerate)


def mix(run: Run):
    from utils.mix import mix_dub
    c = run.config
    mix_dub(read_json(os.path.join(run.fitted_dir, "timing_report.json")), run.vocals, run.background,
            run.duration(), run.mix_dir, target_lufs=c["MIX_TARGET_LUFS"], true_peak=c["MIX_TRUE_PEAK"],
            duck_db=c["MIX_DUCK_DB"], keep_vocals_db=c["MIX_KEEP_VOCALS_DB"])


def mux(run: Run):
    from utils.mux import mux_video
    c = run.config
    mux_video(run.video, os.path.join(run.mix_dir, "dubbed_audio.wav"), read_json(run.translation), run.output_dir,
              keep_original_audio=c["MUX_KEEP_ORIGINAL_AUDIO"], original_subtitles=c["MUX_ORIGINAL_SUBTITLES"],
              audio_bitrate=c["MUX_AUDIO_BITRATE"])


STAGES = [
    Stage("extract", "reading video metadata and extracting the audio", extract,
          lambda r: [r.metadata, r.source_audio]),
    Stage("separate", "separating vocals from background", separate,
          lambda r: [r.vocals, r.background], ["DEMUCS_MODEL", "DEMUCS_SHIFTS", "SEED"]),
    Stage("transcribe", "transcribing the vocals", transcribe_stage,
          lambda r: [r.transcript], ["ASR_BACKEND", "WHISPER_MODEL", "SOURCE_LANGUAGE", "GLOSSARY"]),
    Stage("diarize", "finding who speaks when", diarize_stage,
          lambda r: [r.diarization], ["NUM_SPEAKERS", "MIN_SPEAKERS", "MAX_SPEAKERS"]),
    Stage("segment", "building dubbing segments", segment, lambda r: [r.segments],
          ["SEGMENT_MIN_DURATION", "SEGMENT_MERGE_GAP"]),
    Stage("voices", "picking a voice reference for each speaker", voices, lambda r: [r.references]),
    Stage("translate", "translating the segments", translate, lambda r: [r.translation],
          ["TRANSLATOR", "TARGET_LANGUAGE", "GLOSSARY", "VIDEO_CONTEXT", "LLM_MODEL", "LLM_BASE_URL",
           "NLLB_MODEL"], per_translator=True),
    Stage("tts", "speaking each segment in its speaker's cloned voice", tts,
          lambda r: [r.clips_doc()],
          ["TTS_BACKEND", "F5_MODEL", "TTS_VERIFY", "TTS_ATTEMPTS", "TTS_ON_MISMATCH", "WHISPER_MODEL",
           "FIT_MAX_SPEED"],
          per_translator=True),
    Stage("fit", "fitting each clip into its time slot", fit,
          lambda r: [os.path.join(r.fitted_dir, "timing_report.json")],
          ["FIT_MIN_SPEED", "FIT_MAX_SPEED", "FIT_HARD_MAX_SPEED", "FIT_MAX_BORROW", "FIT_MAX_RETRANSLATIONS"],
          per_translator=True),
    Stage("mix", "mixing the dubbed speech with the background", mix,
          lambda r: [os.path.join(r.mix_dir, "dubbed_audio.wav")],
          ["MIX_TARGET_LUFS", "MIX_TRUE_PEAK", "MIX_DUCK_DB", "MIX_KEEP_VOCALS_DB"], per_translator=True),
    Stage("mux", "creating the dubbed video", mux,
          lambda r: [output_path_for(r.video, r.config["TARGET_LANGUAGE"], r.output_dir)],
          ["MUX_KEEP_ORIGINAL_AUDIO", "MUX_ORIGINAL_SUBTITLES", "MUX_AUDIO_BITRATE", "TARGET_LANGUAGE"],
          per_translator=True),
]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Translate and dub a video. Finished stages are remembered, so re-running resumes where it "
                    "stopped and only redoes stages whose settings changed.")
    parser.add_argument("input", nargs="?", default="input/source-video.mp4", help="path to the source video")
    parser.add_argument("--src", help="source language code (default: detect)")
    parser.add_argument("--tgt", help="target language code, e.g. en")
    parser.add_argument("--translator", choices=["google", "gpt", "nllb"], help="translation backend")
    parser.add_argument("--tts", choices=["f5"], help="voice cloning backend")
    parser.add_argument("--config", help="settings file (default: ./config.toml if present)")
    parser.add_argument("--work-dir", default="work", help="folder for intermediate files (default: work)")
    parser.add_argument("--output-dir", default="output", help="folder for the dubbed video (default: output)")
    stage_names = [s.name for s in STAGES]
    parser.add_argument("--from-stage", choices=stage_names, help="re-run this stage and every stage after it")
    parser.add_argument("--to-stage", choices=stage_names, help="stop after this stage")
    parser.add_argument("--force", action="store_true", help="re-run every stage")
    parser.add_argument("--clean", action="store_true", help="delete this video's work folder and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="also show third-party output (always written to <work folder>/run.log)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        parser.error(f"input video not found: {args.input}")
    config = load_config(args.config, {"SOURCE_LANGUAGE": args.src, "TARGET_LANGUAGE": args.tgt,
                                       "TRANSLATOR": args.translator, "TTS_BACKEND": args.tts})
    # the glossary's content, not its path, decides whether transcription and translation must run again
    from utils.translate import load_glossary
    config["GLOSSARY"] = load_glossary(config["GLOSSARY_FILE"] or None)
    run = Run(args.input, config, args.work_dir, args.output_dir)

    if args.clean:
        if os.path.exists(run.work_dir):
            shutil.rmtree(run.work_dir)
            print(f"🧹 Deleted {run.work_dir}")
        else:
            print(f"Nothing to delete: {run.work_dir} does not exist")
        return

    os.makedirs(run.work_dir, exist_ok=True)
    restore = output.setup(os.path.join(run.work_dir, "run.log"), verbose=args.verbose)
    try:
        dub(run, config, args)
    except BaseException:
        output.log_exception()
        raise
    finally:
        restore()


def dub(run: Run, config: dict, args):
    print(f"🎬 Dubbing {run.video} → {config['TARGET_LANGUAGE']} (translator: {config['TRANSLATOR']}, "
          f"tts: {config['TTS_BACKEND']}); work folder: {run.work_dir}")
    pipeline = Pipeline(STAGES, run.work_dir, config["TRANSLATOR"])
    pipeline.run(run, config, from_stage=args.from_stage, to_stage=args.to_stage, force=args.force,
                 inputs_id=file_id(run.video))
    if os.path.exists(run.clips_doc()):
        from utils.synthesize import fallback_summary
        lines = fallback_summary(read_json(run.clips_doc()))
        if lines:
            print(f"⚠️ {len(lines)} lines were rephrased or not dubbed:")
            for line in lines:
                print(f"   - {line}")
    if not args.to_stage or args.to_stage == STAGES[-1].name:
        print(f"🎉 Done: {output_path_for(run.video, config['TARGET_LANGUAGE'], run.output_dir)}")


if __name__ == "__main__":
    sys.exit(main())
