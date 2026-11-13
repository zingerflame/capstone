"""Acquire whole public broadcasts and prepare explicitly bounded teacher-review clips.

This module does not auto-invent ChatGPT labels. A teacher must inspect exported
review sheets and provide annotations before a clip becomes training data.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .io import read_json, write_json
from .video import sample_video


def acquire_broadcasts(manifest, output):
    import yt_dlp
    output = Path(output)
    video_dir = output / "videos"
    video_dir.mkdir(parents=True,exist_ok=True)
    inventory = []
    for row in read_json(manifest):
        try:
            options = {"format":"bestvideo[height<=480][vcodec^=avc1]/best[height<=480]/bestvideo[height<=480]",
                "outtmpl":str(video_dir / "%(id)s.%(ext)s"), "noplaylist":True,
                "writeinfojson":True,"noprogress":True,"retries":2,"socket_timeout":30}
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(row["source"],download=False)
                if info.get("channel") != row["channel"] or info.get("duration",0)<3600:
                    raise ValueError("Expected official channel and a broadcast at least one hour long")
                if "full match" not in info.get("title", "").lower():
                    raise ValueError("Upload is not identified as a full match")
                existing = [p for p in video_dir.glob(row["id"]+".*") if p.suffix in (".mp4",".webm",".mkv")]
                if existing:
                    path = existing[0]
                else:
                    info = downloader.extract_info(row["source"],download=True)
                    path = Path(downloader.prepare_filename(info))
            capture = cv2.VideoCapture(str(path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, image = capture.read()
            capture.release()
            if not ok or fps<=0 or abs(frames/fps-info["duration"])>max(30,info["duration"]*.02):
                raise ValueError("Downloaded file failed decoding or full-duration verification")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while data := stream.read(1024*1024):
                    digest.update(data)
            record = {**row,"status":"downloaded", "path":str(path.resolve()),
                      "verified_title":info["title"], "channel_id":info.get("channel_id"),
                      "duration_seconds":info["duration"],"decoded_duration_seconds":frames/fps,
                      "fps":fps,"frames":frames,"resolution":[image.shape[1],image.shape[0]],
                      "bytes":path.stat().st_size,"sha256":digest.hexdigest()}
            print(f"VERIFIED {row['id']} {frames/fps:.1f}s {path.stat().st_size/1e6:.1f}MB",flush=True)
        except Exception as exc:
            record = {**row,"status":"failed","error":str(exc)}
            print(f"FAILED {row['id']}: {exc}",flush=True)
        inventory.append(record)
        write_json(output/"inventory.json",inventory)
    return inventory


def contact_sheet(frames, destination, title, columns=5, tile_width=256):
    tile_height = 144
    rows = (len(frames)+columns-1)//columns
    canvas = np.full((44+rows*(tile_height+24),columns*tile_width,3),20,np.uint8)
    cv2.putText(canvas,title[:150],(8,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    for k,(label,image) in enumerate(frames):
        x,y=(k%columns)*tile_width,44+(k//columns)*(tile_height+24)
        canvas[y:y+tile_height,x:x+tile_width]=cv2.resize(image,(tile_width,tile_height))
        cv2.putText(canvas,label,(x+3,y+tile_height+17),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(destination),canvas)


def scout(inventory_path, output):
    """Sparse spread over a fixed early match region, one sheet per match."""
    output=Path(output)
    for row in read_json(inventory_path):
        if row["status"]!="downloaded":
            continue
        if (output/(row["id"]+".jpg")).exists():
            continue
        capture=cv2.VideoCapture(row["path"])
        frames=[]
        # Every 10 seconds, 3:00 to 7:50. Deterministic; labels do not affect selection.
        for second in range(180,480,10):
            capture.set(cv2.CAP_PROP_POS_MSEC,second*1000)
            ok,image=capture.read()
            if ok:
                frames.append((f"{second}s",image))
        capture.release()
        contact_sheet(frames,output/(row["id"]+".jpg"),row["id"]+" | "+row["title"])


def prepare_review(inventory_path, selections_path, output):
    import imageio_ffmpeg
    output=Path(output)
    inventory={r["id"]:r for r in read_json(inventory_path) if r["status"]=="downloaded"}
    for selection in read_json(selections_path):
        row=inventory[selection["video_id"]]
        identifier=selection["id"]
        directory=output/"clips"/identifier
        if (directory/"clip.json").exists():
            previous = read_json(directory/"clip.json")
            expected = {"id":row["id"], "url":row["source"], "start_seconds":selection["start_seconds"],
                        "duration_seconds":selection["duration_seconds"], "sha256":row["sha256"]}
            if previous.get("broadcast") != expected or previous.get("scoreboard_roi") != selection.get("scoreboard_roi"):
                raise ValueError(f"Changed selection/source for {identifier}; use a fresh output directory")
            continue
        temporary=output/"segments"/(identifier+".mp4")
        temporary.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-y","-ss",str(selection["start_seconds"]),
            "-i",row["path"],"-t",str(selection["duration_seconds"]),"-an","-c:v","libx264","-preset","fast",
            "-crf","18",str(temporary)],check=True)
        clip=sample_video(temporary,directory,{**row,"id":identifier,"scoreboard_roi":selection.get("scoreboard_roi")})
        clip["broadcast"]={"id":row["id"],"url":row["source"],"start_seconds":selection["start_seconds"],
                           "duration_seconds":selection["duration_seconds"],"sha256":row["sha256"]}
        for f in clip["frames"]:
            f["broadcast_timestamp"]=selection["start_seconds"]+f["timestamp"]
        write_json(directory/"clip.json",clip)
        for start in range(0,len(clip["frames"]),30):
            frames=[]
            for f in clip["frames"][start:start+30]:
                frames.append((f"{f['index']} | {f['broadcast_timestamp']:.2f}s",cv2.imread(str(directory/f["image"]))))
            contact_sheet(frames,output/"review"/f"{identifier}_{start:04d}.jpg",identifier+" | FILTERED FRAME INDEX")
        print(f"REVIEW {identifier} {len(clip['frames'])} frames",flush=True)


def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="command",required=True)
    q=s.add_parser("acquire");q.add_argument("--manifest",required=True);q.add_argument("--output",required=True)
    q=s.add_parser("scout");q.add_argument("--inventory",required=True);q.add_argument("--output",required=True)
    q=s.add_parser("review");q.add_argument("--inventory",required=True);q.add_argument("--selections",required=True);q.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="acquire":acquire_broadcasts(a.manifest,a.output)
    elif a.command=="scout":scout(a.inventory,a.output)
    else:prepare_review(a.inventory,a.selections,a.output)


if __name__=="__main__":main()
