import srt
from translators import google, gpt

def get_translator():
    """
    Returns the translator either set via environment variable or defaults to GPT
    """
    import os
    translator = os.getenv("TRANSLATOR", "gpt").lower()
    if translator == "google":
        return google
    elif translator == "gpt":
        return gpt
    else:
        raise ValueError(f"Unknown translator: {translator}. Use 'google' or 'gpt'.")

def translate_subtitles(subtitles_path: str, raw_output_path: str, polished_output_path: str):
    print("📄 Reading input subtitles...")
    with open(subtitles_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    translator = get_translator()
    print(f"🔍 Using translator: {translator.__name__}")
    print("🔤 Translating subtitles...")

    translated_subtitles = translator.translate(subtitles)
    print("💾 Saving raw translations...")

    print("💾 Saving polished translations...")
    with open(polished_output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated_subtitles))

    print(f"✅ Finished! Output saved to {polished_output_path}")
