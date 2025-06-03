# video-translator
A video language translator

Script to run the scripts:

export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

1. Setup a virtual environment:
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Get the audio from the source video file:
```bash
ffmpeg -i ./data/source-video.mp4 -ac 1 -ar 16000 ./data/source-audio.wav
```

4. Run the translate script:
```bash
whisper ./data/source-audio.wav --language Japanese --task transcribe --output_format vtt --output_dir ./data
whisper ./data/source-audio.wav --language Japanese --task transcribe --output_format srt --output_dir ./data
```

5. Translate the subtitles to English:
```bash
# Using GPT
python translate.py ./data/source-audio.srt ./data/source-audio-translated-gpt.srt --backend gpt

# Using Google Cloud API
export GOOGLE_APPLICATION_CREDENTIALS="/Users/prateek/work/github/video-project/project-video-translate.json"
python translate.py ./data/source-audio.srt ./data/source-audio-translated-google.srt --backend google

# Using Argos (local)
python translate.py ./data/source-audio.srt ./data/source-audio-translated-argos.srt --backend argos

# Using unofficial deep_translator
python translate.py ./data/source-audio.srt ./data/source-audio-translated-deepl.srt --backend deep
```

```bash
python translate.py --input ./data/source-audio.srt --raw_output ./data/translated_raw.srt --polished_output ./data/translated_polished.srt
```

5. Lip sync the video with the translated audio:
```bash
git clone https://github.com/ram-lakshmanan/Wav2Lip-m1
cd Wav2Lip-m1
pip install -r requirements.txt
bash download_model.sh
python inference.py --checkpoint_path checkpoints/wav2lip.pth \
  --face input.mp4 --audio english_audio.wav --outfile result.mp4 --pads 0 20 0 0 --cpu

```
