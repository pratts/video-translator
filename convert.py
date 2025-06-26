import os
import shutil
from utils.transcribe import extract_audio, transcribe_to_srt
from utils.translate import translate_subtitles
from utils.audio_segment_generator import generate_audio_segments

class Converter:
    def get_path(self, filename):
        """Get the absolute path of a file in the current directory."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    def __init__(self):
        self.source_video_file = self.get_path('input/source-video.mp4')
        self.extracted_audio_file = self.get_path('data/source-audio.wav')
        self.subtitle_file = self.get_path('data/subtitles.srt')
        self.raw_subtitle_file = self.get_path('data/translated_raw.srt')
        self.polished_subtitle_file = self.get_path('data/translated_polished.srt')
        self.output_audio_file = self.get_path('data/audio-output.wav')
        self.bgm_file = self.get_path('data/htdemucs/source-video/no_vocals.wav')
        self.merged_audio_file = self.get_path('data/merged_audio.wav')
        self.vocal_audio_file = self.get_path('data/htdemucs/source-video/vocals.wav')
        self.merged_subtitle_file = self.get_path('data/merged_subtitles.srt')

        # Paths for directories
        self.open_voice_path = self.get_path('data/processed')
        self.audio_segment_dir = self.get_path('data/segments')
        self.htdemucs_dir = self.get_path('data/htdemucs')
        self.output_dir = self.get_path('output')

    def clear_files(self):
        import os
        files_to_clear = [
            self.extracted_audio_file,
            self.subtitle_file,
            self.raw_subtitle_file,
            self.polished_subtitle_file,
            self.output_audio_file,
            self.htdemucs_dir,
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
        extract_audio(self.source_video_file, self.extracted_audio_file)

        print("Step 2: Transcribing audio to SRT...")
        transcribe_to_srt(self.extracted_audio_file, self.subtitle_file, model_size="turbo")

        print("Step 3: Translating subtitles...")
        translate_subtitles(self.subtitle_file, self.raw_subtitle_file, self.polished_subtitle_file)

        print("Step 4: Generating audio segments from vocals...")
        generate_audio_segments(self.polished_subtitle_file, self.vocal_audio_file, self.audio_segment_dir)

        # print("Step 6: Cleaning up temporary files...")
        # self.clear_files()

        print("Conversion process completed successfully!")

if __name__ == "__main__":
    converter = Converter()
    converter.convert()
    print("All steps completed successfully!")
    print("You can now use the output audio file for your video.")