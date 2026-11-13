"""Official GolfDB metadata/sample bootstrap and preprocessed-video manifest adapter."""
import hashlib
import urllib.request
from pathlib import Path
from .io import write_json

UPSTREAM = "https://raw.githubusercontent.com/wmcnally/golfdb/master/"


def bootstrap(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for remote, name in (("data/golfDB.mat", "golfDB.mat"), ("test_video.mp4", "test_video.mp4")):
        path = output / name
        if not path.exists():
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(UPSTREAM+remote,timeout=120) as response, temporary.open("wb") as target:
                while block := response.read(1024*1024):
                    target.write(block)
            temporary.replace(path)
    write_json(output / "sample_manifest.json", [{"id":"golfdb_demo", "source":"test_video.mp4",
        "sport":"golf", "group_id":"golfdb_upstream_demo", "split":"test", "scoreboard_roi":None,
        "sha256":hashlib.sha256((output/"test_video.mp4").read_bytes()).hexdigest()}])
    write_json(output / "source.json", {"repository":"https://github.com/wmcnally/golfdb",
        "paper":"https://arxiv.org/abs/1903.06528", "files":{
            name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ("golfDB.mat","test_video.mp4")},
        "note":"Upstream demonstration video is not a train/val/test benchmark; full videos are separate."})
    return output / "sample_manifest.json"


def make_manifest(mat_path, videos_dir, output, limit=None):
    """Read MATLAB metadata, without loading a remote Python pickle.

    Uses official 160x160 trimmed clips named <id>.mp4. Group by original YouTube
    source and hash into 80/10/10 splits. This is our leakage-resistant baseline
    split, not GolfDB's four-fold published benchmark. Official split retained.
    """
    from scipy.io import loadmat
    rows = loadmat(mat_path, simplify_cells=True)["golfDB"]
    result = []
    for row in rows:
        identifier = int(row["id"])
        path = Path(videos_dir).resolve() / f"{identifier}.mp4"
        if not path.exists():
            continue
        youtube_id = str(row["youtube_id"])
        bucket = int(hashlib.sha256(youtube_id.encode()).hexdigest()[:8],16) % 10
        split = "test" if bucket==0 else "val" if bucket==1 else "train"
        result.append({"id":f"golfdb_{identifier}", "source":str(path), "sport":"golf",
            "group_id":f"golfdb_source_{youtube_id}", "split":split, "scoreboard_roi":None,
            "golfdb_official_fold":int(row["split"]), "golfdb_slow_motion":bool(row["slow"]),
            "golfdb_original_event_frames":[int(x) for x in row["events"]],
            "golfdb_note":"Cropped single-swing baseline, not full-broadcast scoreboard validation"})
        if limit is not None and len(result)>=limit:
            break
    if not result:
        raise ValueError("No matching <id>.mp4 clips; obtain the official videos_160 archive first")
    write_json(output,result)
    return result
