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

def merge_srt_blocks(polished_srt_path: str, output_merged_path: str):
    with open(polished_srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    merged_subs = []
    i = 0
    while i < len(subtitles):
        current = subtitles[i]
        merged_content = [current.content]
        start_time = current.start
        end_time = current.end

        # Merge while end == next.start
        while i + 1 < len(subtitles) and end_time == subtitles[i + 1].start:
            i += 1
            next_sub = subtitles[i]
            merged_content.append(next_sub.content)
            end_time = next_sub.end

        merged_sub = srt.Subtitle(index=len(merged_subs) + 1,
                                  start=start_time,
                                  end=end_time,
                                  content=". ".join(merged_content))
        merged_subs.append(merged_sub)
        i += 1

    with open(output_merged_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(merged_subs))
