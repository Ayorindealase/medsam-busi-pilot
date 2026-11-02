import argparse
import json
from .data import prepare


def main():
    parser = argparse.ArgumentParser(description="BUSI MedSAM pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    for command in ("cache", "train", "evaluate"):
        p = commands.add_parser(command)
        for name in ("root", "manifest", "checkpoint", "output"):
            p.add_argument("--" + name, required=True)
        p.add_argument("--device", default="cuda")
        if command != "cache":
            p.add_argument("--cache", required=True)
        if command == "train":
            p.add_argument("--epochs", type=int, default=10)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--jitter", type=int, default=10)
            p.add_argument("--accumulation", type=int, default=4)
        if command == "evaluate":
            p.add_argument("--decoder")
            p.add_argument("--split", choices=["val", "test"], default="test")
    p = commands.add_parser("compare")
    p.add_argument("--baseline-path", required=True)
    p.add_argument("--tuned-path", required=True)
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["destination"] = args.pop("output")
        result = prepare(**args)
        print(json.dumps({"size": result["size"], "seed": result["seed"]}))
    else:
        from .experiment import cache_embeddings, train, evaluate, compare
        {"cache": cache_embeddings, "train": train, "evaluate": evaluate, "compare": compare}[command](**args)


if __name__ == "__main__":
    main()
