# video-translator
A video language translator

Note: This project is in its early stages and is not yet functional. It is intended to be a proof of concept for a video language translator.

Steps to setup and run the project:
### Prerequisites: ###
- Python 3.12 or higher
- ffmpeg
- Google Cloud account with the Translation API enabled
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

### Setup Instructions: ###
# Video Translator Setup Instructions
# This project uses a virtual environment to manage dependencies and requires specific packages to be installed.
# Follow the steps below to set up the project environment and run the translation service.
1. Setup a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Download the Google application credentials JSON file from your Google Cloud account and place it in a folder. For example, if you save it as a file named `key.json`, the path would be `{folder path}/key.json`.
   - Make sure to enable the Translation API in your Google Cloud project.

4. Export the Google application credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="{folder path}/key.json"
```

5. Extrac background audio from the video:
```bash
python3 -m demucs --two-stems=vocals -o ./data ./data/source-video.mp4

6. Run the convert script to get the translated audio:
```bash
python convert.py
```

7. Merge the translated audio with the original video:
```bash
ffmpeg -i ./data/source-video.mp4 -i ./data/merged_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest ./data/output-video.mp4
```
