import os
import srt
import openai
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

def translate(subtitles, batch_size=10):
    polished = []

    for i in range(0, len(subtitles), batch_size):
        batch = subtitles[i:i+batch_size]
        lines = [s.content for s in batch]

        prompt = dedent(f"""
        Rewrite the following lines into natural, spoken English for dubbing.
        Make them sound like casual dialogue but preserve the meaning.
        Separate each rewritten line with a newline in the same order.

        Original lines:
        {chr(10).join(f'{idx+1}. {line}' for idx, line in enumerate(lines))}
        """)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        outputs = response.choices[0].message.content.strip().splitlines()

        # Clean output and map back
        for orig, new_line in zip(batch, outputs):
            # Remove leading numbering if GPT added it
            line = new_line.strip()
            if line and line[0].isdigit() and line[1] in ['.', ')']:
                line = line[2:].strip()
            polished.append(srt.Subtitle(index=orig.index, start=orig.start, end=orig.end, content=line))

    return polished
