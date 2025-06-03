from deep_translator import GoogleTranslator
import srt

with open("input.srt", "r", encoding="utf-8") as f:
    subtitles = list(srt.parse(f.read()))

translated_subs = []
for sub in subtitles:
    print(sub)
    translated_text = GoogleTranslator(source='ja', target='en').translate(sub.content)
    translated_subs.append(srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=translated_text))

with open("translated.srt", "w", encoding="utf-8") as f:
    f.write(srt.compose(translated_subs))
