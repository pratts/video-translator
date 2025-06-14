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

### Setup the translation service using ffmpeg, whisper and bark audio model.: ###
3. Create a google cloud project and enable the translation API.

4. Export the Google application credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/prateek/work/github/video-project/project-video-translate.json"
```

5. Run the convert script to get the translated audio:
```bash
python convert.py
```

8. Merge the translated audio with the original video:
```bash
ffmpeg -i ./data/source-video.mp4 -i ./data/merged_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest ./data/output-video.mp4
```


brew install rubberband 