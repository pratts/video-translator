import importlib
import json
import os
import srt
from datetime import timedelta

TRANSLATORS = {"google": "translators.google", "gpt": "translators.gpt", "nllb": "translators.nllb"}

# Characters per second of natural dubbed speech (spaces and punctuation included),
# used to give each line a length budget for its time slot
CHARS_PER_SECOND = {
    "en": 14, "es": 14, "fr": 14, "it": 14, "pt": 14, "de": 13, "nl": 13, "pl": 12, "ru": 12, "uk": 12,
    "hi": 12, "bn": 12, "ar": 11, "tr": 12, "id": 13, "vi": 11, "ko": 7, "ja": 7, "zh": 4.5, "yue": 4.5,
}
DEFAULT_CHARS_PER_SECOND = 12
MIN_CHARS = 4


def get_translator(name: str):
    """Import the translator backend on demand, so unused backends need no credentials or packages."""
    name = name.lower()
    if name not in TRANSLATORS:
        raise ValueError(f"Unknown translator: {name}. Use one of: {', '.join(TRANSLATORS)}.")
    return importlib.import_module(TRANSLATORS[name])


def char_budget(duration: float, language: str) -> int:
    return max(MIN_CHARS, int(duration * CHARS_PER_SECOND.get(language, DEFAULT_CHARS_PER_SECOND)))


def load_glossary(path: str | None) -> dict:
    """Glossary JSON: {"source term": "translation"}, e.g. names that must be kept consistent."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# === Step 4: Translate dubbing segments ===
def translate_segments(segments_doc: dict, translation_output_path: str, raw_srt_path: str,
                       polished_srt_path: str, translator: str = "google", target_language: str = "en",
                       glossary: dict | None = None, context: str | None = None) -> dict:
    """
    Translate every dubbing segment into the target language.
    Each segment gets a character budget ("max_chars") from its duration, so LLM backends can
    write lines that fit the time slot; "fits" marks whether the translation is within budget.
    Saves a JSON document for later stages, plus raw (literal) and polished (dubbing) SRTs.
    """
    source_language = segments_doc["language"]
    items = [{"id": s["id"], "speaker": s["speaker"], "duration": s["duration"], "text": s["text"],
              "max_chars": char_budget(s["duration"], target_language)} for s in segments_doc["segments"]]

    backend = get_translator(translator)
    print(f"🔤 Translating {len(items)} segments {source_language} → {target_language} with {translator}...")
    results = backend.translate(items, source_language, target_language, glossary=glossary, context=context) \
        if items else {}

    missing = [item["id"] for item in items if item["id"] not in results]
    if missing:
        raise RuntimeError(f"Translator returned no translation for segment ids {missing}")

    # LLM backends get one more chance to shorten lines that ran over their budget
    if hasattr(backend, "shorten"):
        for item in items:
            result = results[item["id"]]
            if len(result["polished"]) > item["max_chars"]:
                result["polished"] = backend.shorten(item, result["polished"], item["max_chars"], source_language,
                                                     target_language, glossary=glossary, context=context)

    segments = []
    for seg, item in zip(segments_doc["segments"], items):
        result = results[item["id"]]
        segments.append({
            "id": seg["id"],
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "gap_after": seg["gap_after"],
            "source_text": seg["text"],
            "raw": result["raw"],
            "translation": result["polished"],
            "max_chars": item["max_chars"],
            "fits": len(result["polished"]) <= item["max_chars"],
        })

    document = {
        "source_language": source_language,
        "target_language": target_language,
        "translator": translator,
        "duration": segments_doc.get("duration"),
        "speakers": segments_doc.get("speakers", []),
        "glossary": glossary or {},
        "context": context or "",
        "segments": segments,
    }
    save_translation(document, translation_output_path, raw_srt_path, polished_srt_path)
    return document


def shorten(document: dict, segment_id: int, max_chars: int, translator: str | None = None) -> str:
    """
    Rewrite one segment's translation to at most max_chars characters (for the timing-fit stage,
    when a dubbed clip is too long). Needs an LLM backend; updates the document in place.
    """
    backend = get_translator(translator or document["translator"])
    if not hasattr(backend, "shorten"):
        raise NotImplementedError(f"Translator '{backend.__name__}' cannot shorten lines; use TRANSLATOR=gpt")
    seg = next(s for s in document["segments"] if s["id"] == segment_id)
    item = {"id": seg["id"], "speaker": seg["speaker"], "duration": seg["duration"], "text": seg["source_text"]}
    text = backend.shorten(item, seg["translation"], max_chars, document["source_language"],
                           document["target_language"], glossary=document.get("glossary"),
                           context=document.get("context"))
    seg["translation"] = text
    seg["max_chars"] = max_chars
    seg["fits"] = len(text) <= max_chars
    return text


def rephrase(document: dict, segment_id: int, heard: str, translator: str | None = None) -> str:
    """
    Rewrite one segment's translation in plainer words the TTS can pronounce (for the TTS stage, when no
    attempt at the line was heard correctly). Needs an LLM backend; updates the document in place.
    """
    backend = get_translator(translator or document["translator"])
    if not hasattr(backend, "rephrase"):
        raise NotImplementedError(f"Translator '{backend.__name__}' cannot rephrase lines; use TRANSLATOR=gpt")
    seg = next(s for s in document["segments"] if s["id"] == segment_id)
    item = {"id": seg["id"], "speaker": seg["speaker"], "duration": seg["duration"], "text": seg["source_text"]}
    text = backend.rephrase(item, seg["translation"], heard, seg["max_chars"], document["source_language"],
                            document["target_language"], glossary=document.get("glossary"),
                            context=document.get("context"))
    seg["translation"] = text
    seg["fits"] = len(text) <= seg["max_chars"]
    return text


def save_translation(document: dict, json_output_path: str, raw_srt_path: str, polished_srt_path: str):
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    for path, field in [(raw_srt_path, "raw"), (polished_srt_path, "translation")]:
        subtitles = [srt.Subtitle(index=s["id"] + 1, start=timedelta(seconds=s["start"]),
                                  end=timedelta(seconds=s["end"]), content=s[field])
                     for s in document["segments"]]
        with open(path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))

    over = [s["id"] for s in document["segments"] if not s["fits"]]
    print(f"✅ Saved {len(document['segments'])} translations to {json_output_path}, "
          f"{raw_srt_path} and {polished_srt_path}")
    if over:
        print(f"⚠️ {len(over)} segments are longer than their time slot allows: ids {over}")
