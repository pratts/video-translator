from google.cloud import translate_v2 as translate
import srt, os

client = translate.Client()

def translate(subtitles):
    texts = [s.content for s in subtitles]
    results = client.translate(texts, source_language="ja", target_language="en")

    translated = []
    for orig, res in zip(subtitles, results):
        translated.append(srt.Subtitle(index=orig.index, start=orig.start, end=orig.end, content=res["translatedText"]))

    return translated
