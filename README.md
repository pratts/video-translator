# video-translator
Dub a video into another language with the speakers' own (cloned) voices: transcribe, translate, clone each voice, fit the speech to the original timing, mix it with the original background and write the dubbed video, all with one command.

Note: This project is a proof of concept.

## Prerequisites
- Python 3.12 or higher
- ffmpeg and ffprobe on your `PATH` (`brew install ffmpeg`)
- [Rubber Band](https://breakfastquay.com/rubberband/) command-line tool for time-stretching dubbed clips without changing pitch (`brew install rubberband` on macOS, `apt install rubberband-cli` on Debian/Ubuntu). Without it, a lower-quality librosa fallback is used
- A Hugging Face account and [access token](https://huggingface.co/settings/tokens), with the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) accepted (speaker diarization)
- For translation, one of:
  - `google`: a Google Cloud project with the Cloud Translation API and billing enabled, and a service account JSON key
  - `gpt`: an OpenAI API key, or a local OpenAI-compatible server such as [Ollama](https://ollama.com)
  - `nllb`: nothing (fully local)

Models (Demucs, Whisper, wav2vec2, pyannote, F5-TTS, NLLB) are downloaded on first use.

## Installation
```bash
git clone git@github.com:pratts/video-translator.git
cd video-translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```
`constraints.txt` holds the versions tested together; WhisperX, pyannote and F5-TTS only work with compatible `torch`, `transformers` and `huggingface_hub` releases.

## Configuration
1. Copy `.env.example` to `.env` and fill in the secrets: `HF_TOKEN`, and `OPENAI_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` for your translator.
2. Optionally copy `config.example.toml` to `config.toml` and change settings there. Every setting is documented in that file.

Settings are resolved in this order, later ones winning: built-in defaults, `config.toml` (or `--config path`), environment variables / `.env` with the same name in upper case (e.g. `TRANSLATOR="gpt"`), command-line flags.

## Usage
```bash
python convert.py my-video.mp4 --tgt en                     # dub into English
python convert.py my-video.mp4 --tgt es --translator gpt    # pick the translator
python convert.py                                           # uses input/source-video.mp4 and config.toml / .env
```
The dubbed video is written to `output/<video name>.<target language>.mp4` (`.mkv` for MKV input).

### Resuming and re-running stages
Each video gets a work folder, `work/<video name>/`, and every finished stage is recorded in its `state.json`. Running the same command again skips finished stages, so:
- after a failure or interruption, the next run resumes at the stage that did not finish
- after changing a setting, only the stage that uses it and the stages after it run again (e.g. changing `MIX_DUCK_DB` re-runs only `mix` and `mux`)
- stages from `translate` on are kept per translator, so switching between `gpt`, `google` and `nllb` reuses each one's earlier results

Runs are repeatable: the same video and settings give the same separation, transcript, segments and voice clips (`seed` fixes Demucs' random shifts; F5-TTS is seeded per attempt). Only the `gpt` translator can word lines differently between runs.

```bash
python convert.py my-video.mp4 --from-stage translate    # re-run translation and everything after it
python convert.py my-video.mp4 --to-stage segment        # stop after building the segments (e.g. to review them)
python convert.py my-video.mp4 --force                   # re-run everything
python convert.py my-video.mp4 --clean                   # delete the video's work folder
python convert.py my-video.mp4 --verbose                 # also show the libraries' own output
```
Run `python convert.py --help` for all options.

The console shows the pipeline's own progress, warnings and errors. The libraries' own output (model loading details, known harmless warnings, info logs) is left out unless you pass `--verbose`, but everything, including errors, is written to `work/<video name>/run.log`.

### Stages
| Stage | What it does | Main output in `work/<video name>/` |
|---|---|---|
| `extract` | Reads the video metadata and extracts the audio | `metadata.json`, `source-audio.wav` |
| `separate` | Separates vocals from music and effects with [Demucs](https://github.com/adefossez/demucs) | `separation/vocals.wav`, `separation/background.wav` |
| `transcribe` | Transcribes the vocals with [WhisperX](https://github.com/m-bain/whisperX) and aligns every word | `transcript.json`, `subtitles.srt` |
| `diarize` | Finds who speaks when with pyannote | `diarization.json` |
| `segment` | Builds sentence-level, single-speaker dubbing segments | `segments.json`, `segments.srt` |
| `voices` | Picks a clean voice reference for each speaker | `voices/<speaker>.wav`, `voices/references.json` |
| `translate` | Translates every segment, with a length budget from its duration | `translations/<translator>/translation.json`, `translated_raw.srt`, `translated_polished.srt` |
| `tts` | Speaks every segment in its speaker's cloned voice ([F5-TTS](https://github.com/SWivid/F5-TTS)), checked with speech recognition | `translations/<translator>/clips/` |
| `fit` | Fits every clip into its time slot (trim, time-stretch, borrow silence, shorten the line) | `translations/<translator>/fitted/timing_report.json` |
| `mix` | Mixes the dubbed speech with the original background, ducking it under speech, and normalizes loudness | `translations/<translator>/mix/dubbed_audio.wav` |
| `mux` | Writes the dubbed video with the original audio and subtitles as extra, language-tagged tracks | `output/<video name>.<target>.mp4` |

### Translators
- `google`: Google Cloud Translation. Plain machine translation, not shortened for dubbing.
- `gpt`: an LLM writes a literal translation and a shorter, natural dubbing line that fits the time slot, using neighbouring lines, `video_context` and the glossary. Set `llm_base_url` to use a local OpenAI-compatible server, e.g. `http://localhost:11434/v1` for Ollama with `llm_model = "qwen2.5:14b"`. Only `gpt` can shorten lines that are too long during `fit`.
- `nllb`: fully local [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M). No account needed, lower quality.

`glossary_file` is an optional JSON file of names and terms that must be translated consistently, e.g. `{"熊谷真奈": "Mana Kumagai"}`. Its source terms are also given to Whisper as a prompt, so names are transcribed with the right spelling (熊谷真奈, not a same-sounding 熊谷モナ); this works with every translator. Editing the glossary re-runs transcription and everything after it.

### Lines the voice can't say
Every clip is transcribed again and accepted only when it says its line: at most 20% of the words differ and no content word is missing, heard as a different word, or added (F5 sometimes speaks words of the voice reference before the line). Clips whose speech, without the silence F5 adds at its edges, covers less than 90% of a slot of 1 s or more are generated again with more room, so lines don't end early. A failing line is generated again with other seeds (`tts_attempts`), then rephrased once in plainer words (`gpt` only). If it still fails, `tts_on_mismatch` decides: `original` keeps the original speaker's voice for that line, `silence` drops it, `keep` uses the closest attempt, and `auto` (default) keeps the original voice for lines up to 1 s (short replies like はい) and drops longer ones. The run ends with a list of the lines that were rephrased or not dubbed; `clips.json` has the details.

### Output video
The video stream is copied without re-encoding and the dubbed audio is the default track. The source audio (copied as is) and original-language subtitles are added as extra tracks, and all tracks are tagged with their language. MP4 players show the first subtitle track by default (the MP4 muxer always enables it); in MKV, subtitles are off by default.

## Tests
```bash
pytest                          # unit tests, about 2 s, no models or network needed
RUN_E2E=1 pytest -m e2e -s      # full pipeline on a 12 s cut of input/source-video.mp4 with real models (local nllb translator, needs HF_TOKEN)
```

## Licenses
The code is MIT. Some models have non-commercial licenses: the pretrained F5-TTS `F5TTS_v1_Base` weights and NLLB-200 are CC-BY-NC 4.0, so dubbed output made with them is for non-commercial use only.
