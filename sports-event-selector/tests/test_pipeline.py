import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
import pytest
import torch
from sports_events.demo import create_demo
from sports_events.io import read_json, write_json
from sports_events.learning import discover, clip_logits
from sports_events.model import Selector
from sports_events.stream import FrameStream
from sports_events.taxonomy import SPORTS, sport_name
from sports_events.vlm import VLM, annotate_clip, generate_taxonomy


def test_video_sampling_split_integrity_and_model_gradient(tmp_path):
    torch.set_num_threads(2)
    dataset = create_demo(tmp_path)
    records = discover(dataset)
    directory, clip, _ = records[0]
    assert 20 <= len(clip["frames"]) <= 43
    assert [f["source_frame"] for f in clip["frames"][:2]] == [0,1]
    assert [f["source_frame"] for f in clip["frames"][-2:]] == [118,119]
    model = Selector()
    logits, _, _ = clip_logits(model,directory,clip,"cpu",batch_size=8)
    logits.sum().backward()
    assert model.scene.layers[0].weight.grad.abs().sum() > 0
    assert model.board.layers[0].weight.grad.abs().sum() > 0
    other_path = dataset / "synthetic_val/clip.json"
    other = read_json(other_path)
    other["group_id"] = "synthetic_match_0"
    write_json(other_path,other)
    with pytest.raises(ValueError,match="leaks"):
        discover(dataset)


def test_stream_matches_batch_on_identical_pixels(tmp_path):
    torch.set_num_threads(2)
    dataset = create_demo(tmp_path)
    directory = dataset / "synthetic_test"
    clip = read_json(directory / "clip.json")
    # Batch loader reads lossy scoreboard JPEGs; disable the board to compare identical inputs.
    clip["scoreboard_roi"] = None
    model = Selector().eval()
    stream = FrameStream(model,.2)
    for f in clip["frames"]:
        stream.push(cv2.imread(str(directory/f["image"])), f["timestamp"])
    with torch.no_grad():
        logits, _, _ = clip_logits(model,directory,clip,"cpu",batch_size=7)
    assert np.allclose(np.stack(stream.probabilities),logits.sigmoid().numpy(),atol=1e-6)
    result = stream.finish()
    assert result["forced_indices"] == [0,1,len(clip["frames"])-2,len(clip["frames"])-1]


def test_vlm_http_contract_taxonomy_cache_and_review_preservation(tmp_path,monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            content = body["messages"][0]["content"]
            if "nine sport keys" in content[0]["text"]:
                result = SPORTS
            else:
                result = {"events":[{"frame_indices":[4,5,6],"label":"player passes", "confidence":.9}]}
            payload = json.dumps({"choices":[{"message":{"content":json.dumps(result)}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self,*args):
            pass
    server = HTTPServer(("127.0.0.1",0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    monkeypatch.setenv("VLM_ENDPOINT",f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setenv("VLM_MODEL","mock-vision-model")
    monkeypatch.setenv("VLM_PROVIDER","compatible")
    try:
        client = VLM()
        taxonomy = generate_taxonomy(client,tmp_path/"taxonomy.json")
        dataset = create_demo(tmp_path/"fixture")
        directory = dataset/"synthetic_train"
        (directory/"annotations.json").unlink()
        annotation = annotate_clip(directory,client,taxonomy,chunk_size=100)
        assert annotation["events"][0]["frame_indices"] == [4,5,6]
        assert len(requests)==2
        assert any(c["type"]=="image_url" for c in requests[-1]["messages"][0]["content"])
        annotation["reviewed"] = True
        write_json(directory/"annotations.json",annotation)
        assert annotate_clip(directory,client,taxonomy,chunk_size=100)["reviewed"]
        assert len(requests)==2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_sport_aliases():
    assert sport_name("Batminton") == "badminton"
    assert sport_name("PingPong") == "table_tennis"
    assert len(SPORTS)==9
