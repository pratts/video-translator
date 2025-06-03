import os
import time
import openai
import srt
from dotenv import load_dotenv

# Load .env and set OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate(subtitles, batch_size=5):
    translated = []
    batches = [subtitles[i:i + batch_size] for i in range(0, len(subtitles), batch_size)]

    for batch in batches:
        # Join all subtitle lines as one prompt
        text = "\n".join(f"{s.index}: {s.content}" for s in batch)
        print(f"Translating batch of {len(batch)} subtitles...")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a subtitle translator. Translate the following Japanese subtitles to English. "
                        "Keep each line's index intact (e.g., '1: ...', '2: ...') in your response."
                    )
                },
                {"role": "user", "content": text}
            ],
        )

        translated_lines = response.choices[0].message.content.strip().split("\n")

        for orig, line in zip(batch, translated_lines):
            try:
                translated_text = line.split(":", 1)[1].strip()
            except IndexError:
                translated_text = orig.content
            translated.append(srt.Subtitle(index=orig.index, start=orig.start, end=orig.end, content=translated_text))

        time.sleep(1.5)

    return translated
