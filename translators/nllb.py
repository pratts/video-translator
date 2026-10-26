import os
from functools import cache

BATCH_SIZE = 8

# ISO 639-1 codes (as detected by Whisper) to NLLB's FLORES-200 codes.
# Any other FLORES-200 code (e.g. "mar_Deva") can be passed as is.
FLORES_CODES = {
    "ar": "arb_Arab", "bn": "ben_Beng", "de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn",
    "fr": "fra_Latn", "hi": "hin_Deva", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "nl": "nld_Latn", "pl": "pol_Latn", "pt": "por_Latn", "ru": "rus_Cyrl",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl",
    "ur": "urd_Arab", "vi": "vie_Latn", "yue": "yue_Hant", "zh": "zho_Hans",
}


def _flores(language: str) -> str:
    if "_" in language:
        return language
    if language not in FLORES_CODES:
        raise ValueError(f"No NLLB code for language '{language}'. Use a FLORES-200 code such as 'eng_Latn'.")
    return FLORES_CODES[language]


@cache
def _load():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = os.getenv("NLLB_MODEL") or "facebook/nllb-200-distilled-600M"
    device = os.getenv("TRANSLATION_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔍 Loading {name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    return tokenizer, model, device


def translate(items: list[dict], source_language: str, target_language: str,
              glossary: dict | None = None, context: str | None = None) -> dict[int, dict]:
    """
    Fully local machine translation with NLLB-200. Returns {id: {"raw", "polished"}}; both are the
    same, because plain machine translation does not adapt lines for dubbing. Glossary and context are ignored.
    """
    import torch

    tokenizer, model, device = _load()
    tokenizer.src_lang = _flores(source_language)
    target_token = tokenizer.convert_tokens_to_ids(_flores(target_language))

    results = {}
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        inputs = tokenizer([item["text"] for item in batch], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, forced_bos_token_id=target_token, num_beams=4,
                                     max_new_tokens=int(inputs["input_ids"].shape[1] * 3) + 10)
        for item, text in zip(batch, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
            results[item["id"]] = {"raw": text.strip(), "polished": text.strip()}
    return results
