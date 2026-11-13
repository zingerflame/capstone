import argparse
import json
from pathlib import Path
from .io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Scoreboard-aware sports event selector")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("golfdb-bootstrap", help="Download official metadata and small golf demo video")
    p.add_argument("--output", default="data/golfdb")
    p = sub.add_parser("golfdb-manifest", help="Create grouped manifest from official preprocessed GolfDB clips")
    p.add_argument("--metadata", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("taxonomy", help="Generate all nine sports' event vocabularies using the VLM")
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare", help="Download and sample videos sequentially")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--taxonomy")
    p.add_argument("--keep-downloads", action="store_true")
    p = sub.add_parser("annotate", help="Annotate an already prepared clip")
    p.add_argument("--clip", required=True)
    p.add_argument("--taxonomy", required=True)
    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-unreviewed", action="store_true")
    p = sub.add_parser("predict", help="Return event frame arrays without text labels")
    p.add_argument("--clip", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("demo", help="Generate synthetic video, train, predict and evaluate")
    p.add_argument("--output", default="runs/demo")
    p.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.command == "golfdb-bootstrap":
        from .golfdb import bootstrap
        print(bootstrap(args.output))
    elif args.command == "golfdb-manifest":
        from .golfdb import make_manifest
        rows = make_manifest(args.metadata,args.videos,args.output,args.limit)
        print(f"Wrote {len(rows)} GolfDB clips to {args.output}")
    elif args.command == "taxonomy":
        from .vlm import VLM, generate_taxonomy
        generate_taxonomy(VLM(), args.output)
    elif args.command == "prepare":
        from .video import prepare
        callback = None
        if args.annotate:
            if not args.taxonomy:
                parser.error("--annotate requires --taxonomy")
            from .vlm import VLM, annotate_clip
            client, taxonomy = VLM(), read_json(args.taxonomy)
            callback = lambda directory: annotate_clip(directory, client, taxonomy)
        print(json.dumps(prepare(args.manifest, args.output, callback, args.keep_downloads), indent=2))
    elif args.command == "annotate":
        from .vlm import VLM, annotate_clip
        annotate_clip(args.clip, VLM(), read_json(args.taxonomy))
    elif args.command == "train":
        from .learning import train
        train(args.dataset, args.output, args.epochs, args.device, args.allow_unreviewed)
    elif args.command == "predict":
        from .learning import load_model, infer_clip
        model, state = load_model(args.checkpoint, args.device)
        clip = read_json(Path(args.clip) / "clip.json")
        write_json(args.output, infer_clip(model, args.clip, clip, args.device, state["budget_rate"],
                                          state["threshold"], state["consensus"]))
    elif args.command == "evaluate":
        from .learning import evaluate
        evaluate(args.dataset, args.checkpoint, args.output, args.split, args.device)
    elif args.command == "demo":
        import torch
        from .demo import create_demo
        from .learning import train, load_model, infer_clip, evaluate
        torch.set_num_threads(2)
        output = Path(args.output)
        dataset = create_demo(output)
        train(dataset, output / "model", args.epochs)
        model, state = load_model(output / "model/best.pt")
        directory = dataset / "synthetic_test"
        result = infer_clip(model, directory, read_json(directory / "clip.json"), "cpu", state["budget_rate"])
        write_json(output / "prediction.json", result)
        evaluate(dataset, output / "model/best.pt", output / "evaluation.json")
        print(f"Synthetic integration demo complete: {output.resolve()}")


if __name__ == "__main__":
    main()
