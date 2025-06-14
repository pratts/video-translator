# from transcribe import extract_audio, transcribe_to_srt
# from translate import translate_subtitles
from speech import convert_srt_to_audio_with_timeline
from speech_clone import clone_speech_with_subtitles
class Converter:
    def __init__(self):
        self.source_video_file = './data/source-video.mp4'
        self.extracted_audio_file = './data/source-audio.wav'
        self.subtitle_file = './data/subtitles.srt'
        self.raw_subtitle_file = './data/translated_raw.srt'
        self.polished_subtitle_file = './data/translated_polished.srt'
        self.output_audio_file = './data/audio-output2.wav'
        self.htdemucs = './data/htdemucs'

    def clear_files(self):
        import os
        files_to_clear = [
            self.extracted_audio_file,
            self.subtitle_file,
            self.raw_subtitle_file,
            self.polished_subtitle_file,
            self.output_audio_file,
            self.htdemucs
        ]
        for file in files_to_clear:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleared file: {file}")
            else:
                print(f"File not found, skipping: {file}")

    def convert(self):
        print("Starting conversion process...")
        print("Step 1: Extracting audio from video...")
        extract_audio(self.source_video_file, self.extracted_audio_file)

        print("Step 2: Transcribing audio to SRT...")
        transcribe_to_srt(self.extracted_audio_file, self.subtitle_file, model_size="turbo")

        # print("Step 3: Translating subtitles...")
        translate_subtitles(self.subtitle_file, self.raw_subtitle_file, self.polished_subtitle_file)

        # print("Step 4: Converting polished subtitles to audio with timeline...")
        # convert_srt_to_audio_with_timeline(self.polished_subtitle_file, self.output_audio_file)
        clone_speech_with_subtitles(self.extracted_audio_file, self.polished_subtitle_file, self.output_audio_file)
        print("Step 5: Cloning speech with subtitles...")

        # self.clear_files()
        
        print("Conversion process completed successfully!")

if __name__ == "__main__":
    converter = Converter()
    converter.convert()
    print("All steps completed successfully!")
    print("You can now use the output audio file for your video.")