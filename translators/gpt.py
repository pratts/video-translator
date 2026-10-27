import json
import os
from functools import cache
from textwrap import dedent

# Lines translated per request, and neighbouring lines sent with them as read-only context
BATCH_SIZE = 20
CONTEXT_LINES = 3
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = dedent("""
    You translate video dialogue for dubbing. Each line will be spoken by a voice actor in the
    same time slot as the original, so it must sound like natural spoken {target} and fit its slot.

    Rules:
    - Return exactly one translation for every id in "lines", and nothing for the context lines.
    - "literal_translation": a faithful, complete {target} translation of the line (never the original text).
    - "dubbing_line": what the voice actor says, in {target}. Natural, conversational, same meaning and
      tone as the speaker, and at most "max_chars" characters, so it can be spoken in "duration" seconds.
      Drop filler words and repetition before dropping meaning.
    - Lines can be fragments of a sentence that continues in the next line, sometimes from
      another speaker. Keep each line's content in that line so the sequence reads naturally.
    - Use the glossary translations for names and terms.
    - Never merge, split, reorder or skip lines.
""").strip()

TRANSLATION_SCHEMA = {
    "name": "dubbing_translations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "literal_translation": {"type": "string"},
                        "dubbing_line": {"type": "string"},
                    },
                    "required": ["id", "literal_translation", "dubbing_line"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}

SHORTEN_SCHEMA = {
    "name": "shortened_line",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
}


@cache
def _client():
    # created on first use, so OPENAI_API_KEY is only needed when this backend is selected.
    # LLM_BASE_URL points it at any OpenAI-compatible server, e.g. Ollama: http://localhost:11434/v1
    from openai import OpenAI
    base_url = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY") or ("unused" if base_url else None)
    return OpenAI(api_key=api_key, base_url=base_url)


def _model() -> str:
    return os.getenv("LLM_MODEL") or "gpt-5-mini"


def _complete(system: str, payload: dict, schema: dict) -> dict:
    response = _client().chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"Model refused the request: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise RuntimeError("Model response was cut off (finish_reason=length); lower BATCH_SIZE")
    return json.loads(choice.message.content)


def _line(item: dict) -> dict:
    return {"id": item["id"], "speaker": item["speaker"], "duration": item["duration"],
            "max_chars": item["max_chars"], "text": item["text"]}


def translate(items: list[dict], source_language: str, target_language: str,
              glossary: dict | None = None, context: str | None = None) -> dict[int, dict]:
    """
    Translate items in batches with neighbouring lines as context, using structured JSON output.
    Every id is checked; ids missing from a reply are retried. Returns {id: {"raw", "polished"}}.
    """
    system = SYSTEM_PROMPT.format(target=target_language)
    results = {}
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        before = items[max(0, start - CONTEXT_LINES):start]
        after = items[start + BATCH_SIZE:start + BATCH_SIZE + CONTEXT_LINES]

        pending = {item["id"]: item for item in batch}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            payload = {
                "source_language": source_language,
                "target_language": target_language,
                "video_context": context or "",
                "glossary": glossary or {},
                "context_before": [{"speaker": i["speaker"], "text": i["text"]} for i in before],
                "lines": [_line(item) for item in pending.values()],
                "context_after": [{"speaker": i["speaker"], "text": i["text"]} for i in after],
            }
            reply = _complete(system, payload, TRANSLATION_SCHEMA)
            for t in reply["translations"]:
                if t["id"] in pending and t["dubbing_line"].strip():
                    results[t["id"]] = {"raw": t["literal_translation"].strip(), "polished": t["dubbing_line"].strip()}
                    del pending[t["id"]]
            if not pending:
                break
            print(f"⚠️ Missing translations for ids {sorted(pending)} (attempt {attempt}/{MAX_ATTEMPTS})")
        if pending:
            raise RuntimeError(f"No translation returned for ids {sorted(pending)}")
    return results


def shorten(item: dict, translation: str, max_chars: int, source_language: str, target_language: str,
            glossary: dict | None = None, context: str | None = None) -> str:
    """Rewrite one translated line to at most max_chars characters, keeping its meaning."""
    system = dedent(f"""
        You shorten dubbed {target_language} dialogue so it fits its time slot.
        Rewrite "translation" in at most {max_chars} characters as natural spoken {target_language},
        keeping the meaning of "original" ({source_language}). Use the glossary for names and terms.
    """).strip()
    payload = {"original": item["text"], "translation": translation, "speaker": item["speaker"],
               "max_chars": max_chars, "video_context": context or "", "glossary": glossary or {}}

    best = translation
    for _ in range(MAX_ATTEMPTS):
        if len(best) <= max_chars:
            break
        text = _complete(system, payload, SHORTEN_SCHEMA)["text"].strip()
        if text and len(text) < len(best):
            best = text
        payload["translation"] = best
    return best


def rephrase(item: dict, translation: str, heard: str, max_chars: int, source_language: str,
             target_language: str, glossary: dict | None = None, context: str | None = None) -> str:
    """Reword a dubbed line that the voice synthesizer could not pronounce clearly, keeping its meaning."""
    system = dedent(f"""
        You fix dubbed {target_language} dialogue for a text-to-speech voice.
        The voice could not say "translation" clearly: speech recognition heard "heard" instead.
        Rewrite it as natural spoken {target_language} with the same meaning as "original" ({source_language}),
        using common, easy-to-pronounce words in place of the ones that were misheard, written out as they
        are spoken (no abbreviations, symbols or digits). At most {max_chars} characters.
        Keep names and glossary terms exactly as they are written in "translation", even if they were misheard:
        never respell a name phonetically (not "Mah nah" for "Mana"). Only reword the other words.
    """).strip()
    payload = {"original": item["text"], "translation": translation, "heard": heard, "speaker": item["speaker"],
               "max_chars": max_chars, "video_context": context or "", "glossary": glossary or {}}
    text = _complete(system, payload, SHORTEN_SCHEMA)["text"].strip()
    return text or translation
