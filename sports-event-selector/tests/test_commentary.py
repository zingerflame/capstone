import json
from pathlib import Path

import pytest

from sports_events.commentary import Provider, prepare, finish, sequence_prompt
from sports_events.io import write_json, read_json


def fixture_data(tmp_path):
    dataset, predictions, output = (tmp_path / x for x in ('data', 'predictions', 'out'))
    clip = dataset / 'clip'
    clip.mkdir(parents=True)
    frames = []
    for i in range(12):
        (clip / f'{i}.jpg').write_bytes(b'test-image')
        frames.append({'image': f'{i}.jpg', 'source_frame': 100+i, 'broadcast_timestamp': 10+i/10})
    write_json(clip / 'clip.json', {'id': 'clip', 'split': 'test', 'group_id': 'tenniset_V010', 'frames': frames})
    write_json(clip / 'annotations.json', {'events': [
        {'sequence_id': 'truth', 'frame_indices': [2, 3, 4], 'label': 'SECRET player hits a shot'}]})
    write_json(predictions / 'clip.json', {'event_sequences': [[2, 3], [4, 5]], 'selected_indices': [0, 1, 2, 3, 4, 5, 10, 11]})
    manifest, _ = prepare(dataset, predictions, 'V010', output)
    obs = [{'id': e['id'], 'kind': 'hit', 'text': 'A player swings.', 'confidence': .7} for e in manifest['sequences']]
    return output, manifest, obs


def test_no_label_leak_and_one_to_one(tmp_path):
    output, manifest, obs = fixture_data(tmp_path)
    assert 'SECRET' not in json.dumps(manifest)
    assert 'SECRET' not in sequence_prompt(obs, manifest)
    segments = [{'id': e['id'], 'text': e['text']} for e in obs]
    result = finish(output, obs, segments, {'mode': 'test'})
    assert len(result['typed_vlm_matches']) == 1
    assert result['typed_precision'] == .5
    assert result['typed_recall'] == 1
    assert read_json(output / 'commentary.json')['segments'][0]['start'] == 10.2


def test_missing_or_reordered_events_rejected(tmp_path):
    output, manifest, obs = fixture_data(tmp_path)
    with pytest.raises(ValueError):
        finish(output, obs[::-1], [], {})
    with pytest.raises(ValueError):
        finish(output, obs, [{'id': obs[0]['id'], 'text': 'Only one'}], {})


@pytest.mark.parametrize('provider', ['gpt', 'gemini'])
def test_provider_payload_cache_and_usage(tmp_path, monkeypatch, provider):
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-test-key')
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-test-key')
    image = tmp_path / 'input.jpg'; image.write_bytes(b'jpeg-test')
    calls = []
    answer = {'kind': 'uncertain', 'text': 'Waiting.', 'confidence': .3}
    response = ({'status': 'completed', 'model': 'test', 'usage': {'input_tokens': 10},
                 'output': [{'content': [{'type': 'output_text', 'text': json.dumps(answer)}]}]}
                if provider == 'gpt' else {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(answer)}]}}],
                                         'usageMetadata': {'promptTokenCount': 10}})
    import io
    def fake(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr('urllib.request.urlopen', fake)
    client = Provider(provider, 'test-model', tmp_path)
    assert client.ask('Return JSON', [image]) == answer
    assert client.ask('Return JSON', [image]) == answer
    assert len(calls) == 1
    if provider == 'gpt':
        assert calls[0]['input'][0]['content'][1]['type'] == 'input_image'
    else:
        assert calls[0]['contents'][0]['parts'][1]['inlineData']['mimeType'] == 'image/jpeg'
    stored = next((tmp_path / 'calls').glob('*.json')).read_text()
    assert 'fake-test-key' not in stored
    assert 'usage' in stored
