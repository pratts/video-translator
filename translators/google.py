from functools import cache

# Google Translation API v2 accepts at most 128 texts per request
MAX_TEXTS_PER_REQUEST = 128


@cache
def _client():
    # created on first use, so GOOGLE_APPLICATION_CREDENTIALS is only needed when this backend is selected
    from google.cloud import translate_v2
    return translate_v2.Client()


def translate(items: list[dict], source_language: str, target_language: str,
              glossary: dict | None = None, context: str | None = None) -> dict[int, dict]:
    """
    Machine-translate each item's text. Returns {id: {"raw", "polished"}}; both are the same,
    because plain machine translation does not adapt lines for dubbing.
    Glossary and context are not supported by the v2 API and are ignored.
    """
    results = {}
    for i in range(0, len(items), MAX_TEXTS_PER_REQUEST):
        batch = items[i:i + MAX_TEXTS_PER_REQUEST]
        # format_="text" stops the API from HTML-escaping quotes and apostrophes
        response = _client().translate([item["text"] for item in batch], source_language=source_language,
                                       target_language=target_language, format_="text")
        if len(response) != len(batch):
            raise RuntimeError(f"Google returned {len(response)} translations for {len(batch)} texts")
        for item, result in zip(batch, response):
            text = result["translatedText"].strip()
            results[item["id"]] = {"raw": text, "polished": text}
    return results
