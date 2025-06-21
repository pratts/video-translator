# video-translator
A video language translator

Note: This project is in its early stages and is not yet functional. It is intended to be a proof of concept for a video language translator.

Steps to setup and run the project:
### Prerequisites: ###
- Python 3.12 or higher
- ffmpeg
- Google Cloud account with the Translation API enabled if using Google Translation
- ChatGPT API key (if using ChatGPT for translation)
- `demucs` installed (for audio extraction)
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

3. Create a `data` directory to store the source video and output files:
```bash
mkdir data
```

4. Place your source video file in the `data` directory.
5. Update the source video, temporary file paths and output files in the `convert.py` script as needed.

### Follow the steps below to set up the project environment and run the translation service. ###
1. Setup a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

3.  Rename the `.env.example` file to `.env` and set the following parameters:
```
TRANSLATOR = google/gpt based on your preference
CHATGPT_API_KEY = your_chatgpt_api_key
GOOGLE_APPLICATION_CREDENTIALS = {folder path}/key.json
```
- `TRANSLATOR` can be set to either `google` or `gpt`:
   - If you choose `google`, the script will use the Google Translation API. Set the `GOOGLE_APPLICATION_CREDENTIALS` variable to the path of your Google application credentials JSON file.
   - If you choose `gpt`, it will use the ChatGPT API for translation. Set the `CHATGPT_API_KEY` variable with your ChatGPT API key.

4. Export the Google application credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="{folder path}/key.json"
```

5. Extrac background audio from the video:
```bash
python3 -m demucs --two-stems=vocals -o ./data ./data/source-video.mp4
```

6. Run the convert script to get the translated audio:
```bash
python convert.py
```

7. Merge the translated audio with the original video:
```bash
ffmpeg -i ./data/source-video.mp4 -i ./data/merged_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest ./data/output-video.mp4
```
