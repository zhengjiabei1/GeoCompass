"""Compare online policies at two stream lengths.

Usage::

    python scripts/compare_horizons.py \
        --short experiments/geocompass/results.json \
        --long experiments/horizon16/results.json

The short run is the default protocol; the long run repeats it with more stream
tasks. Both aggregate blocks store a per-step cumulative-regret curve, so the
comparison reads the curve at the same checkpoint instead of re-running
anything. The marginal regret of the extra steps is the number that matters: if
it falls, exploration is beginning to pay for itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def curve_at(curve: list[float], step: int) -> float:
    if not curve:
        return 0.0
    index = min(step, len(curve)) - 1
    return curve[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", default="experiments/geocompass/results.json")
    parser.add_argument("--long", default="experiments/horizon16/results.json")
    args = parser.parse_args(argv)

    short = json.loads(Path(args.short).read_text())["aggregate"]["bandit"]
    long = json.loads(Path(args.long).read_text())["aggregate"]["bandit"]
    shared = [name for name in short if name in long]
    short_len = min(len(short[name]["curve"]) for name in shared)
    long_len = min(len(long[name]["curve"]) for name in shared)

    header = (
        f"| Policy | Regret @ {short_len} | Regret @ {long_len} | "
        f"Marginal regret per step | Success @ {long_len} | Negative transfer @ {long_len} |"
    )
    print(header)
    print("|---|---:|---:|---:|---:|---:|")
    for name in shared:
        early = curve_at(long[name]["curve"], short_len)
        late = curve_at(long[name]["curve"], long_len)
        extra = max(1, long_len - short_len)
        print(
            f"| {name} | {early:.2f} | {late:.2f} | {(late - early) / extra:.4f} | "
            f"{long[name]['success']:.3f} | {long[name]['negative_transfer']:.3f} |"
        )
    print()
    print(f"short stream {short_len} steps, long stream {long_len} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
