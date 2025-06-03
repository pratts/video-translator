# video-translator
A video language translator

Script to run the scripts:

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
ffmpeg -i input.mp4 -ac 1 -ar 16000 input.wav
```

4. Run the translate script:
```bash
whisper input.wav --language Japanese --task transcribe --output_format vtt
whisper input.wav --language Japanese --task transcribe --output_format srt
```

5. Translate the subtitles to English:
```bash
# Using GPT
python translate.py input.srt output.srt --backend gpt

# Using Google Cloud API
python translate.py input.srt output.srt --backend google

# Using Argos (local)
python translate.py input.srt output.srt --backend argos

# Using unofficial deep_translator
python translate.py input.srt output.srt --backend deep
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
