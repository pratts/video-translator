# video-translator
A video language translator

Note: This project is in its early stages and is not yet functional. It is intended to be a proof of concept for a video language translator.

Steps to setup and run the project:
### Prerequisites: ###
- Python 3.12 or higher
- ffmpeg and ffprobe on your `PATH` (used for audio extraction and reading video metadata)
- Google Cloud account with the Translation API enabled if using Google Translation
- ChatGPT API key (if using ChatGPT for translation)
- [Demucs](https://github.com/adefossez/demucs) >= 4.1.0 (installed through `requirements.txt`; used to separate vocals from background audio)
- `rubberband` and `mecab` installed (for audio processing)

### Installation Instructions: ###
1. Clone the repository:
```bash
git clone git@github.com:pratts/video-translator.git
```

2. Change into the project directory:
```bash
cd video-translator
```

3. Place your source video file in the `input` directory.

### Follow the steps below to prepare the project for translation. ###
1. Setup a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Run the clear command to delete the temp folders. This will ensure that any previous temporary files are removed before starting a new translation.:
```bash
python clear_files.py
```

4.  Rename the `.env.example` file to `.env` and set the following parameters:
```
TRANSLATOR = google/gpt based on your preference
CHATGPT_API_KEY = your_chatgpt_api_key
GOOGLE_APPLICATION_CREDENTIALS = {folder path}/key.json
DEMUCS_MODEL = htdemucs_ft
DEMUCS_DEVICE =
ASR_BACKEND = whisperx
WHISPER_MODEL = large-v3-turbo
SOURCE_LANGUAGE =
ASR_DEVICE =
```
- `TRANSLATOR` can be set to either `google` or `gpt`:
   - If you choose `google`, the script will use the Google Translation API. Set the `GOOGLE_APPLICATION_CREDENTIALS` variable to the path of your Google application credentials JSON file.
   - If you choose `gpt`, it will use the ChatGPT API for translation. Set the `CHATGPT_API_KEY` variable with your ChatGPT API key.
- `DEMUCS_MODEL` is the Demucs model used for vocal separation. `htdemucs_ft` gives the best quality; `htdemucs` is about 4x faster.
- `DEMUCS_DEVICE` can be `cuda`, `mps` or `cpu`. Leave it empty to use CUDA if available, otherwise CPU.
- `ASR_BACKEND` is the speech recognition backend. Both are followed by [WhisperX](https://github.com/m-bain/whisperX) word alignment:
   - `whisperx` (default): faster-whisper with voice activity detection. Most accurate text, but long segments (up to 30s).
   - `mlx`: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), Apple Silicon only. Fast, with shorter natural segments, but less accurate on names.
- `WHISPER_MODEL` is the Whisper model, e.g. `large-v3-turbo`, `large-v3`, `medium`.
- `SOURCE_LANGUAGE` is the spoken language code (e.g. `ja`). Leave it empty to auto-detect.
- `ASR_DEVICE` can be `cuda` or `cpu` (whisperx backend). Leave it empty to use CUDA if available, otherwise CPU.

5. Export the Google application credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="{folder path}/key.json"
```

6. Run the convert script to get the translated audio. It extracts the audio, saves the video metadata to `data/metadata.json`, separates vocals from background into `data/separation/`, then transcribes the vocals to `data/transcript.json` (language, segments and word-level timestamps) and `data/subtitles.srt`:
```bash
python convert.py                      # uses input/source-video.mp4
python convert.py path/to/my-video.mp4 # or any other video
```

7. Clone and update the paths in OpenVoice repository at https://github.com/pratts/OpenVoice.
   - Follow the instructions in the repository.
   - Run the script `python openvoice_clone.py` to generate cloned audio segments.

8. In the file `merge_cloned_audio.py`, update the paths of the input data and OpenVoice output path and run the script:
```bash
python merge_cloned_audio.py
```

9. Merge the translated audio with the original video:
```bash
ffmpeg -i ./input/source-video.mp4 -i ./data/merged_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest ./output/output-video.mp4
```
