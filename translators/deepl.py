import srt
from deep_translator import GoogleTranslator

def translate(subtitles):
    return [
        srt.Subtitle(index=s.index, start=s.start, end=s.end,
                     content=GoogleTranslator(source="ja", target="en").translate(s.content))
        for s in subtitles
    ]
