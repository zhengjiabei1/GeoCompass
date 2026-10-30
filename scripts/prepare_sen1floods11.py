"""Convert a Sen1Floods11 download into a GeoCompass chip directory.

The archive is not redistributed here.  Download the ``HandLabeled`` subset (or a
subset of it) from the collection's public location, then:

    python scripts/prepare_sen1floods11.py \
        --images data/raw/S2Hand \
        --labels data/raw/LabelHand \
        --out data/sen1floods11 \
        --family flood \
        --split-map splits.json

``splits.json`` maps chip ids to ``source_train`` / ``source_val`` /
``target_dev`` / ``target_test`` / ``stream``.  Without it every chip goes to
``--default-split``.

The script never downloads anything and never writes outside ``--out``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geocompass.prepare import (  # noqa: E402
    Chip,
    SEN1FLOODS11_BANDS,
    convert_image,
    convert_labels,
    manifest_payload,
    write_manifest,
)

IMAGE_SUFFIX = "_S2Hand.tif"
LABEL_SUFFIX = "_LabelHand.tif"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory with *_S2Hand.tif")
    parser.add_argument("--labels", required=True, help="directory with *_LabelHand.tif")
    parser.add_argument("--out", required=True, help="destination for chips and manifest")
    parser.add_argument("--family", default="flood")
    parser.add_argument("--default-split", default="target_test")
    parser.add_argument("--split-map", default=None, help="JSON object: chip id to split")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="convert at most N chips")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    images = sorted(Path(args.images).glob(f"*{IMAGE_SUFFIX}"))
    if not images:
        print(f"no {IMAGE_SUFFIX} files under {args.images}", file=sys.stderr)
        return 1
    if args.limit:
        images = images[: args.limit]
    split_map = json.loads(Path(args.split_map).read_text()) if args.split_map else {}
    out = Path(args.out)
    chips: list[Chip] = []
    skipped = 0
    for image in images:
        identifier = image.name[: -len(IMAGE_SUFFIX)]
        label = Path(args.labels) / f"{identifier}{LABEL_SUFFIX}"
        if not label.exists():
            print(f"skip {identifier}: no matching label", file=sys.stderr)
            skipped += 1
            continue
        split = split_map.get(identifier, args.default_split)
        if args.dry_run:
            print(f"{identifier}\t{split}\t{label.name}")
            continue
        image_out = convert_image(image, out / f"{identifier}.tif", SEN1FLOODS11_BANDS)
        label_out = convert_labels(label, out / f"{identifier}_label.tif")
        chips.append(Chip(identifier=identifier, image=image_out.name,
                          labels=label_out.name, family=args.family, split=split))
    if args.dry_run:
        print(f"{len(images) - skipped} chips ready, {skipped} skipped")
        return 0
    if not chips:
        print("nothing converted", file=sys.stderr)
        return 1
    payload = manifest_payload(chips, name=out.name, size=args.size)
    write_manifest(payload, out / "manifest.json")
    print(f"wrote {len(chips)} chips and {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
