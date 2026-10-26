import json

import httpx
import openai
import pytest

from translators import gpt
from utils import translate as T


def segments_doc(n):
    return {"language": "ja", "duration": n * 2.0, "speakers": ["S"],
            "segments": [{"id": i, "speaker": "S", "start": i * 2.0, "end": i * 2.0 + 1.5, "duration": 1.5,
                          "gap_after": 0.5, "text": f"line {i}", "words": []} for i in range(n)]}


@pytest.fixture
def fake_openai(monkeypatch):
    """OpenAI client backed by a local handler; calls records every request body."""
    calls = []

    def install(reply):
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            content = reply(json.loads(body["messages"][1]["content"]), len(calls))
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(content), "refusal": None}}]})
        client = openai.OpenAI(api_key="test", base_url="http://test/v1",
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        monkeypatch.setattr(gpt, "_client", lambda: client)
        return calls
    return install


def test_every_id_is_matched_and_missing_ids_are_retried(fake_openai, tmp_path, monkeypatch):
    monkeypatch.setattr(gpt, "BATCH_SIZE", 50)

    def reply(payload, call):
        ids = [line["id"] for line in payload["lines"]]
        out = [{"id": i, "literal_translation": f"lit {i}", "dubbing_line": f"dub {i}"} for i in ids]
        if call == 1:  # drop one id, blank one, add an unknown one, and shuffle
            out = [{**o, "dubbing_line": ""} if o["id"] == 5 else o for o in out if o["id"] != 3][::-1]
            out.append({"id": 999, "literal_translation": "x", "dubbing_line": "x"})
        return {"translations": out}

    calls = fake_openai(reply)
    doc = T.translate_segments(segments_doc(120), str(tmp_path / "t.json"), str(tmp_path / "raw.srt"),
                               str(tmp_path / "pol.srt"), translator="gpt", target_language="en")
    assert [s["translation"] for s in doc["segments"]] == [f"dub {i}" for i in range(120)]
    assert [s["raw"] for s in doc["segments"]] == [f"lit {i}" for i in range(120)]
    retried = [line["id"] for line in json.loads(calls[1]["messages"][1]["content"])["lines"]]
    assert retried == [3, 5]
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_gives_up_when_an_id_never_comes_back(fake_openai, tmp_path):
    fake_openai(lambda payload, call: {"translations": []})
    with pytest.raises(RuntimeError, match="No translation returned"):
        T.translate_segments(segments_doc(2), str(tmp_path / "t.json"), str(tmp_path / "r.srt"),
                             str(tmp_path / "p.srt"), translator="gpt")


def test_shorten_returns_the_shorter_line(fake_openai):
    fake_openai(lambda payload, call: {"text": "Short."})
    doc = {"translator": "gpt", "source_language": "ja", "target_language": "en", "glossary": {}, "context": "",
           "segments": [{"id": 0, "speaker": "S", "duration": 1.0, "source_text": "x",
                         "translation": "A much longer line than allowed", "max_chars": 30, "fits": False}]}
    assert T.shorten(doc, 0, 10) == "Short."
    assert doc["segments"][0]["fits"] is True


def test_google_batches_and_keeps_order(monkeypatch, tmp_path):
    from translators import google

    class FakeClient:
        def __init__(self):
            self.batches = []

        def translate(self, texts, source_language, target_language, format_):
            self.batches.append(len(texts))
            return [{"translatedText": t.upper()} for t in texts]

    client = FakeClient()
    monkeypatch.setattr(google, "_client", lambda: client)
    doc = T.translate_segments(segments_doc(300), str(tmp_path / "t.json"), str(tmp_path / "r.srt"),
                               str(tmp_path / "p.srt"), translator="google")
    assert client.batches == [128, 128, 44]
    assert doc["segments"][299]["translation"] == "LINE 299"


def test_char_budget_scales_with_duration_and_language():
    assert T.char_budget(2.0, "en") == 28
    assert T.char_budget(2.0, "ja") == 14
    assert T.char_budget(0.1, "en") == T.MIN_CHARS
