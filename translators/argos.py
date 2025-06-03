import srt
from argostranslate import package, translate
import os

def translate(subtitles):
    installed_languages = translate.load_installed_languages()
    from_lang = next(lang for lang in installed_languages if lang.code == "ja")
    to_lang = next(lang for lang in installed_languages if lang.code == "en")
    translator = from_lang.get_translation(to_lang)

    return [
        srt.Subtitle(index=s.index, start=s.start, end=s.end, content=translator.translate(s.content))
        for s in subtitles
    ]
