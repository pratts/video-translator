import srt
import argparse
from translators import google, gpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.srt", help="Input SRT file")
    parser.add_argument("--raw_output", default="translated_raw.srt")
    parser.add_argument("--polished_output", default="translated_polished.srt")
    args = parser.parse_args()

    print("📄 Reading input subtitles...")
    with open(args.input, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    print("🔤 Translating with Google Translate...")
    translated = google.translate(subtitles)

    print("💾 Saving raw translations...")
    with open(args.raw_output, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated))

    print("🧠 Polishing translations with GPT...")
    polished = gpt.polish_subtitles(translated)

    print("💾 Saving polished translations...")
    with open(args.polished_output, "w", encoding="utf-8") as f:
        f.write(srt.compose(polished))

    print(f"✅ Finished! Output saved to {args.polished_output}")

if __name__ == "__main__":
    print("Starting translation process...")
    main()
