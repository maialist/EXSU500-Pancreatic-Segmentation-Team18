"""Create a deterministic validation proxy/confirmation manifest.

The manifest keeps the original training split and divides only the original
validation patients. Its ``val`` list is used for proxy screening and its
``test`` list is used for confirmation; neither list is the actual test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Split validation patients into deterministic proxy and confirmation groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/random_forest_proxy_split.json"),
    )
    parser.add_argument("--proxy-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Preserve the shared training and actual test sets; subdivide validation only.
    with args.splits.open(encoding="utf-8") as handle:
        source = json.load(handle)
    validation_cases = list(source["val"])
    if not 1 <= args.proxy_cases < len(validation_cases):
        raise ValueError("--proxy-cases must be between 1 and len(validation)-1")

    rng = np.random.default_rng(args.seed)
    selected = set(
        rng.choice(validation_cases, size=args.proxy_cases, replace=False).tolist()
    )
    proxy = [case for case in validation_cases if case in selected]
    confirmation = [case for case in validation_cases if case not in selected]
    # The experiment's "test" key means confirmation, not the held-out project test.
    manifest = {
        "train": list(source["train"]),
        "val": proxy,
        "test": confirmation,
        "actual_test": list(source["test"]),
        "metadata": {
            "source_splits": str(args.splits),
            "proxy_cases": len(proxy),
            "confirmation_cases": len(confirmation),
            "seed": args.seed,
            "note": "The test key is confirmation patients for experiments, not the held-out project test split.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(proxy)} proxy and {len(confirmation)} confirmation cases to {args.out}")


if __name__ == "__main__":
    main()
