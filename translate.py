import srt
import argparse
from translators import google, gpt

def translate_subtitles(subtitles_path: str, raw_output_path: str, polished_output_path: str):
    print("📄 Reading input subtitles...")
    with open(subtitles_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    print("🔤 Translating with Google Translate...")
    translated = google.translate(subtitles)

    print("💾 Saving raw translations...")
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated))

    print("🧠 Polishing translations with GPT...")
    polished = gpt.polish_subtitles(translated)

    print("💾 Saving polished translations...")
    with open(polished_output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(polished))

    print(f"✅ Finished! Output saved to {polished_output_path}")
