import argparse
import json
import os
import shutil
from dotenv import load_dotenv

# Load .env before importing the pipeline modules, so settings are available when they load
load_dotenv()

from utils.media import extract_audio, probe_media
from utils.separate import separate_vocals
from utils.transcribe import transcribe
from utils.diarize import diarize
from utils.segment import build_segments, save_segments
from utils.translate import translate_segments, load_glossary
from utils.voices import build_voice_references
from utils.synthesize import synthesize_segments, make_regenerator
from utils.fit import fit_clips
from utils.mix import mix_dub

def env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default

class Converter:
    def get_path(self, filename):
        """Get the absolute path of a file in the current directory."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    def __init__(self, source_video_file: str):
        self.source_video_file = os.path.abspath(source_video_file)
        self.metadata_file = self.get_path('data/metadata.json')
        self.extracted_audio_file = self.get_path('data/source-audio.wav')
        self.asr_audio_file = self.get_path('data/source-audio-16k.wav')
        self.transcript_file = self.get_path('data/transcript.json')
        self.subtitle_file = self.get_path('data/subtitles.srt')
        self.diarization_file = self.get_path('data/diarization.json')
        self.segments_file = self.get_path('data/segments.json')
        self.segments_subtitle_file = self.get_path('data/segments.srt')
        # each translator writes to its own folder, and later stages read from the selected translator's folder
        self.translator = os.getenv("TRANSLATOR", "google").lower()
        self.translation_dir = self.get_path(f'data/translations/{self.translator}')
        self.translation_file = os.path.join(self.translation_dir, 'translation.json')
        self.raw_subtitle_file = os.path.join(self.translation_dir, 'translated_raw.srt')
        self.polished_subtitle_file = os.path.join(self.translation_dir, 'translated_polished.srt')
        self.bgm_file = self.get_path('data/separation/background.wav')
        self.vocal_audio_file = self.get_path('data/separation/vocals.wav')

        # Paths for directories
        self.voices_dir = self.get_path('data/voices')
        self.clips_dir = os.path.join(self.translation_dir, 'clips')
        self.fitted_dir = os.path.join(self.translation_dir, 'fitted')
        self.mix_dir = os.path.join(self.translation_dir, 'mix')
        self.separation_dir = self.get_path('data/separation')
        self.output_dir = self.get_path('output')

    def clear_files(self):
        import os
        files_to_clear = [
            self.metadata_file,
            self.extracted_audio_file,
            self.asr_audio_file,
            self.transcript_file,
            self.subtitle_file,
            self.diarization_file,
            self.segments_file,
            self.segments_subtitle_file,
            self.translation_dir,
            self.separation_dir,
            self.voices_dir,
        ]
        for file in files_to_clear:
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)
                print(f"Cleared file: {file}")
            else:
                print(f"File not found, skipping: {file}")

    def create_directories(self):
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./input', exist_ok=True)
        os.makedirs('./output', exist_ok=True)
    
    def convert(self):
        print("Starting conversion process...")

        print("Step 0: Clearing temporary files...")
        self.create_directories()

        print("Step 1: Extracting audio from video...")
        probe_media(self.source_video_file, self.metadata_file)
        extract_audio(self.source_video_file, self.extracted_audio_file)
        extract_audio(self.source_video_file, self.asr_audio_file, sample_rate=16000, mono=True)

        print("Step 1b: Separating vocals from background...")
        separate_vocals(self.extracted_audio_file, self.vocal_audio_file, self.bgm_file,
                        model=os.getenv("DEMUCS_MODEL", "htdemucs_ft"),
                        device=os.getenv("DEMUCS_DEVICE") or None)

        print("Step 2: Transcribing vocals...")
        transcribe(self.vocal_audio_file, self.transcript_file, self.subtitle_file,
                   backend=os.getenv("ASR_BACKEND", "whisperx"),
                   model_name=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
                   language=os.getenv("SOURCE_LANGUAGE") or None,
                   device=os.getenv("ASR_DEVICE") or None)

        print("Step 3: Finding speakers and building dubbing segments...")
        turns = diarize(self.vocal_audio_file, self.diarization_file, os.getenv("HF_TOKEN"),
                        num_speakers=env_int("NUM_SPEAKERS"),
                        min_speakers=env_int("MIN_SPEAKERS"),
                        max_speakers=env_int("MAX_SPEAKERS"),
                        device=os.getenv("DIARIZATION_DEVICE") or None)
        with open(self.transcript_file, encoding="utf-8") as f:
            transcript = json.load(f)
        with open(self.metadata_file) as f:
            video_duration = json.load(f)["duration"]
        segments = build_segments(transcript, turns, video_duration)
        save_segments(segments, self.segments_file, self.segments_subtitle_file)

        print("Step 4: Translating segments...")
        translation = translate_segments(segments, self.translation_file, self.raw_subtitle_file,
                                         self.polished_subtitle_file,
                                         translator=self.translator,
                                         target_language=os.getenv("TARGET_LANGUAGE") or "en",
                                         glossary=load_glossary(os.getenv("GLOSSARY_FILE")),
                                         context=os.getenv("VIDEO_CONTEXT") or None)

        print("Step 5: Cloning each speaker's voice for the translated segments...")
        references = build_voice_references(segments, turns, self.vocal_audio_file, self.bgm_file, self.voices_dir)
        tts = dict(backend=os.getenv("TTS_BACKEND", "f5"),
                   verify=os.getenv("TTS_VERIFY", "true").lower() != "false",
                   attempts=env_int("TTS_ATTEMPTS", 3),
                   asr_model=os.getenv("WHISPER_MODEL", "large-v3-turbo"))
        clips = synthesize_segments(translation, references, self.clips_dir, **tts)

        print("Step 6: Fitting each clip into its time slot...")
        timing = fit_clips(clips, self.fitted_dir, video_duration=video_duration,
                           min_speed=env_float("FIT_MIN_SPEED", 0.9),
                           max_speed=env_float("FIT_MAX_SPEED", 1.25),
                           hard_max_speed=env_float("FIT_HARD_MAX_SPEED", 1.6),
                           max_borrow=env_float("FIT_MAX_BORROW", 0.5),
                           max_retries=env_int("FIT_MAX_RETRANSLATIONS", 2),
                           regenerate=make_regenerator(translation, self.translation_file, references, **tts))

        print("Step 7: Mixing the dubbed speech with the background...")
        keep_vocals = os.getenv("MIX_KEEP_VOCALS_DB")
        mix_dub(timing, self.vocal_audio_file, self.bgm_file, video_duration, self.mix_dir,
                target_lufs=env_float("MIX_TARGET_LUFS", -16.0),
                true_peak=env_float("MIX_TRUE_PEAK", -1.5),
                duck_db=env_float("MIX_DUCK_DB", -6.0),
                keep_vocals_db=float(keep_vocals) if keep_vocals else None)

        # print("Step 8: Cleaning up temporary files...")
        # self.clear_files()

        print("Conversion process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate and dub a video.")
    parser.add_argument("input", nargs="?", default="input/source-video.mp4", help="Path to the source video")
    args = parser.parse_args()

    converter = Converter(args.input)
    converter.convert()
    print("All steps completed successfully!")
    print("You can now use the output audio file for your video.")