import argparse
import os
import shutil
from dotenv import load_dotenv

# Load .env before importing modules that read it at import time (e.g. translators/google.py)
load_dotenv()

from utils.media import extract_audio, probe_media
from utils.separate import separate_vocals
from utils.transcribe import transcribe
# from utils.translate import translate_subtitles
# from utils.audio_segment_generator import generate_audio_segments

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
        self.raw_subtitle_file = self.get_path('data/translated_raw.srt')
        self.polished_subtitle_file = self.get_path('data/translated_polished.srt')
        self.output_audio_file = self.get_path('data/audio-output.wav')
        self.bgm_file = self.get_path('data/separation/background.wav')
        self.merged_audio_file = self.get_path('data/merged_audio.wav')
        self.vocal_audio_file = self.get_path('data/separation/vocals.wav')
        self.merged_subtitle_file = self.get_path('data/merged_subtitles.srt')

        # Paths for directories
        self.open_voice_path = self.get_path('data/processed')
        self.audio_segment_dir = self.get_path('data/segments')
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
            self.raw_subtitle_file,
            self.polished_subtitle_file,
            self.output_audio_file,
            self.separation_dir,
            self.merged_audio_file,
            self.audio_segment_dir,
            self.open_voice_path,
            self.output_audio_file
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

        # print("Step 3: Translating subtitles...")
        # translate_subtitles(self.subtitle_file, self.raw_subtitle_file, self.polished_subtitle_file)

        # print("Step 4: Generating audio segments from vocals...")
        # generate_audio_segments(self.polished_subtitle_file, self.vocal_audio_file, self.audio_segment_dir)

        # print("Step 6: Cleaning up temporary files...")
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