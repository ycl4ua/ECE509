"""Parse a train.py stdout log into a history.json for plotting.

Runs started before the history patch still have one line per epoch in their
background output file; this script recovers the metrics.
"""

import argparse
import json
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+clean\s+([\d.]+)\s+"
        r"\|\s+attack\s+([\d.]+)\s+\|\s+noise\s+([\d.]+)\s+\|\s+lr\s+([\d.]+)\s+"
        r"\|\s+([\d.]+)s"
    )
    history = []
    with open(args.log, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            history.append(
                {
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(3)),
                    "clean_acc": float(match.group(4)),
                    "attack_acc": float(match.group(5)),
                    "noise_acc": float(match.group(6)),
                    "lr": float(match.group(7)),
                    "seconds": float(match.group(8)),
                }
            )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"history": history}, handle, indent=2)
    print(f"Parsed {len(history)} epochs -> {args.out}")
    if history:
        print("first:", history[0])
        print("last:", history[-1])


if __name__ == "__main__":
    main()
