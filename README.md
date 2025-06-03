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
export GOOGLE_APPLICATION_CREDENTIALS="/Users/prateek/work/github/video-project/project-video-translate.json"
python translate.py --input ./data/subtitles.srt --raw_output ./data/t_raw.srt --polished_output ./data/t_polished.srt
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
