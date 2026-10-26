# video-translator
A video language translator

Note: This project is in its early stages and is not yet functional. It is intended to be a proof of concept for a video language translator.

Steps to setup and run the project:
### Prerequisites: ###
- Python 3.12 or higher
- ffmpeg and ffprobe on your `PATH` (used for audio extraction and reading video metadata)
- Google Cloud account with the Translation API enabled if using Google Translation
- OpenAI API key if using GPT for translation (or a local OpenAI-compatible server such as [Ollama](https://ollama.com))
- Hugging Face account and access token, with the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) accepted (for speaker diarization)
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
DEMUCS_MODEL = htdemucs_ft
DEMUCS_DEVICE =
ASR_BACKEND = whisperx
WHISPER_MODEL = large-v3-turbo
SOURCE_LANGUAGE =
ASR_DEVICE =
HF_TOKEN = your_hugging_face_token
NUM_SPEAKERS =
MIN_SPEAKERS =
MAX_SPEAKERS =
DIARIZATION_DEVICE =
TRANSLATOR = google
TARGET_LANGUAGE = en
GLOSSARY_FILE =
VIDEO_CONTEXT =
GOOGLE_APPLICATION_CREDENTIALS = {folder path}/key.json
OPENAI_API_KEY = your_openai_api_key
LLM_MODEL = gpt-5-mini
LLM_BASE_URL =
NLLB_MODEL = facebook/nllb-200-distilled-600M
TRANSLATION_DEVICE =
```
- `DEMUCS_MODEL` is the Demucs model used for vocal separation. `htdemucs_ft` gives the best quality; `htdemucs` is about 4x faster.
- `DEMUCS_DEVICE` can be `cuda`, `mps` or `cpu`. Leave it empty to use CUDA if available, otherwise CPU.
- `ASR_BACKEND` is the speech recognition backend. Both are followed by [WhisperX](https://github.com/m-bain/whisperX) word alignment:
   - `whisperx` (default): faster-whisper with voice activity detection. Most accurate text, but long segments (up to 30s).
   - `mlx`: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), Apple Silicon only. Fast, with shorter natural segments, but less accurate on names.
- `WHISPER_MODEL` is the Whisper model, e.g. `large-v3-turbo`, `large-v3`, `medium`.
- `SOURCE_LANGUAGE` is the spoken language code (e.g. `ja`). Leave it empty to auto-detect.
- `ASR_DEVICE` can be `cuda` or `cpu` (whisperx backend). Leave it empty to use CUDA if available, otherwise CPU.
- `HF_TOKEN` is a [Hugging Face access token](https://huggingface.co/settings/tokens) used for speaker diarization. Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) with the same account first.
- `NUM_SPEAKERS`, or `MIN_SPEAKERS` / `MAX_SPEAKERS`, can be set if you know how many people speak. Leave them empty to detect automatically.
- `DIARIZATION_DEVICE` can be `cuda`, `mps` or `cpu`. Leave it empty to use CUDA, then the Apple GPU (`mps`), then CPU.
- `TRANSLATOR` is the translation backend. Every segment gets a character budget from its duration, and segments that are too long for their time slot are reported:
   - `google`: Google Cloud Translation. Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your service account JSON key. Plain machine translation, not shortened for dubbing.
   - `gpt`: an LLM writes a literal translation and a shorter, natural dubbing line that fits the time slot, using neighbouring lines, `VIDEO_CONTEXT` and the glossary. Set `OPENAI_API_KEY` and `LLM_MODEL`. To run it locally, set `LLM_BASE_URL` to an OpenAI-compatible server, e.g. `http://localhost:11434/v1` for Ollama, and `LLM_MODEL` to a local model such as `qwen2.5:14b`.
   - `nllb`: fully local [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) machine translation (`NLLB_MODEL`, `TRANSLATION_DEVICE`). No account needed, lower quality; the model license is non-commercial (CC-BY-NC 4.0).
- `TARGET_LANGUAGE` is the language to dub into (e.g. `en`, `es`, `hi`). The source language comes from the transcript.
- `GLOSSARY_FILE` is an optional JSON file of names and terms that must be translated consistently, e.g. `{"熊谷真奈": "Mana Kumagai"}`. `VIDEO_CONTEXT` is an optional one-line description of the video. Both are used by `gpt`.

5. Export the Google application credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="{folder path}/key.json"
```

6. Run the convert script to get the translated audio. It extracts the audio, saves the video metadata to `data/metadata.json`, separates vocals from background into `data/separation/`, transcribes the vocals to `data/transcript.json` (language, segments and word-level timestamps) and `data/subtitles.srt`, finds who speaks when (`data/diarization.json`), builds sentence-level, single-speaker dubbing segments in `data/segments.json` (with `data/segments.srt` for review), and translates them to `data/translation.json` (source text, literal and dubbing translations, character budget per segment) with `data/translated_raw.srt` and `data/translated_polished.srt`:
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
