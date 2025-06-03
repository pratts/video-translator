import argparse
import srt
from translators import gpt, deepl

BACKENDS = {
    "gpt": gpt.translate,
    # "google": google_api.translate,
    # "argos": argos.translate,
    "deep": deepl.translate,
}

def main():
    parser = argparse.ArgumentParser(description="Translate subtitles using various backends.")
    parser.add_argument("input", help="Input SRT file")
    parser.add_argument("output", help="Output SRT file")
    parser.add_argument("--backend", choices=BACKENDS.keys(), default="gpt", help="Translation backend")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    translated = BACKENDS[args.backend](subtitles)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated))

    print(f"✅ Done: {args.backend} -> {args.output}")

if __name__ == "__main__":
    main()
