import pytest
from sports_events.golfdb import make_manifest


def test_golfdb_adapter_groups_source_videos(tmp_path,monkeypatch):
    scipy = pytest.importorskip("scipy.io")
    rows = [{"id":i,"youtube_id":"same_source","split":i+1,"slow":0,"events":[1,2,3]} for i in range(3)]
    monkeypatch.setattr(scipy,"loadmat",lambda *a,**k:{"golfDB":rows})
    for i in range(3):
        (tmp_path/f"{i}.mp4").touch()
    result = make_manifest("unused.mat",tmp_path,tmp_path/"manifest.json")
    assert len(result)==3
    assert len({row["split"] for row in result})==1
    assert len({row["group_id"] for row in result})==1
    assert all(row["scoreboard_roi"] is None for row in result)
