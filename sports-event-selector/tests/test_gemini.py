import io
import json
import cv2
import numpy as np
import pytest
from sports_events.vlm import VLM, generate_taxonomy
from sports_events.taxonomy import SPORTS


def test_native_gemini_multimodal_json_transport(tmp_path,monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER","gemini")
    monkeypatch.setenv("GEMINI_API_KEY","test-only-key")
    monkeypatch.setenv("GEMINI_MODEL","test-gemini-model")
    image = tmp_path/"frame.jpg"
    cv2.imwrite(str(image),np.zeros((32,32,3),np.uint8))
    calls = []
    def urlopen(request,timeout):
        body = json.loads(request.data)
        calls.append((request,body))
        return io.BytesIO(json.dumps({"candidates":[{"finishReason":"STOP","content":{
            "parts":[{"text":json.dumps(SPORTS)}]}}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen",urlopen)
    client = VLM()
    assert client.ask("describe",[("Frame 0",image)],schema={"type":"object"}) == SPORTS
    request, body = calls[0]
    assert request.full_url.endswith("/test-gemini-model:generateContent")
    assert request.get_header("X-goog-api-key") == "test-only-key"
    assert body["contents"][0]["parts"][2]["inlineData"]["mimeType"] == "image/jpeg"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert generate_taxonomy(client,tmp_path/"taxonomy.json") == SPORTS


def test_gemini_missing_key_is_actionable(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER","gemini")
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    with pytest.raises(ValueError,match="Set GEMINI_API_KEY"):
        VLM()


def test_gemini_does_not_accept_truncated_json(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER","gemini")
    monkeypatch.setenv("GEMINI_API_KEY","test-only-key")
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:io.BytesIO(
        b'{"candidates":[{"finishReason":"MAX_TOKENS","content":{"parts":[{"text":"{}"}]}}]}'))
    with pytest.raises(ValueError,match="no complete answer"):
        VLM().ask("describe")
