"""Load a chip collection described by a manifest.

The synthetic generator produces scenes from a task specification.  Real chips
arrive as files, so they need the opposite flow: read the files, then build the
specification.  This module does that from a small JSON manifest, which keeps the
runner, the skill library, the selector, and the report unchanged.

Manifest format::

    {
      "name": "sen1floods11-sample",
      "band_names": ["blue", "green", "red", "nir", "swir1", "swir2"],
      "scale": 0.0001,
      "size": 32,
      "entries": [
        {
          "id": "Bolivia_103757",
          "family": "flood",
          "split": "target_test",
          "pre": ["Bolivia_103757_S2Hand.tif"],
          "labels": "Bolivia_103757_LabelHand.tif",
          "region": "bolivia",
          "season": "wet"
        }
      ]
    }

``pre`` and ``post`` are either a list of per-band paths in ``band_names`` order,
or the same list keyed by band name, or a single multi-band path.  Relative paths
resolve against the manifest directory.  ``scale`` and ``offset`` are applied to
every band, because the correct factors depend on the product level rather than
on the file.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .realdata import SceneSpec, load_scene
from .tasks import build_library
from .types import FAMILIES, Scene, Skill, TaskSpec

SPLITS = ("source_train", "source_val", "target_dev", "target_test", "stream")


class ManifestError(ValueError):
    """Raised when a manifest is malformed or points at missing files."""


@dataclass
class Entry:
    identifier: str
    family: str
    split: str
    pre: Sequence[str]
    post: Sequence[str] | None
    labels: str | None
    region: str = "real"
    season: str = "unknown"
    change_type: str = "mixed"
    resolution: int = 10


def _resolve(root: Path, value: Any, field: str) -> Sequence[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items: Sequence[str] = [value]
    elif isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, list):
        items = value
    else:
        raise ManifestError(f"{field}: expected a path, a list, or a mapping")
    resolved = []
    for item in items:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise ManifestError(f"{field}: missing file {path}")
        resolved.append(str(path))
    return resolved


def read_manifest(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(data, dict) or "entries" not in data:
        raise ManifestError(f"{path}: expected an object with an 'entries' list")
    if not data["entries"]:
        raise ManifestError(f"{path}: 'entries' is empty")
    data["_root"] = path.parent
    return data


def parse_entries(manifest: dict) -> list[Entry]:
    root: Path = manifest["_root"]
    entries = []
    for index, row in enumerate(manifest["entries"]):
        for key in ("id", "family", "split"):
            if key not in row:
                raise ManifestError(f"entry {index}: missing '{key}'")
        if row["family"] not in FAMILIES:
            raise ManifestError(
                f"entry {index}: unknown family {row['family']!r}; expected one of {list(FAMILIES)}")
        if row["split"] not in SPLITS:
            raise ManifestError(
                f"entry {index}: unknown split {row['split']!r}; expected one of {list(SPLITS)}")
        pre = _resolve(root, row.get("pre"), f"entry {index} pre")
        if not pre:
            raise ManifestError(f"entry {index}: 'pre' is required")
        entries.append(Entry(
            identifier=str(row["id"]),
            family=row["family"],
            split=row["split"],
            pre=pre,
            post=_resolve(root, row.get("post"), f"entry {index} post"),
            labels=(str(root / row["labels"]) if row.get("labels") else None),
            region=str(row.get("region", "real")),
            season=str(row.get("season", "unknown")),
            change_type=str(row.get("change_type", "mixed")),
            resolution=int(row.get("resolution", 10)),
        ))
    identifiers = [entry.identifier for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ManifestError("entry ids must be unique")
    return entries


def load_dataset(path: str | Path, seed: int = 0) -> tuple[dict[str, list[TaskSpec]], dict[str, Scene], list[Skill]]:
    """Build splits, scenes, and a skill library from a manifest."""
    manifest = read_manifest(path)
    entries = parse_entries(manifest)
    band_names = manifest.get("band_names")
    if band_names is not None and len(band_names) != 6:
        raise ManifestError("band_names must list six bands in acquisition order")
    spec = SceneSpec(scale=float(manifest.get("scale", 1.0)),
                     offset=float(manifest.get("offset", 0.0)),
                     size=int(manifest["size"]) if manifest.get("size") else None)

    splits: dict[str, list[TaskSpec]] = {name: [] for name in SPLITS}
    scenes: dict[str, Scene] = {}
    for entry in entries:
        task_id = entry.identifier
        scene = load_scene(entry.pre, band_names=band_names, labels_path=entry.labels,
                           post_paths=entry.post, spec=spec)
        task = TaskSpec(
            task_id=task_id,
            family=entry.family,
            split=entry.split,
            seed=seed,
            region=entry.region,
            season=entry.season,
            resolution=entry.resolution,
            size=scene.size,
            change_type=entry.change_type,
        )
        splits[entry.split].append(task)
        scenes[task_id] = scene

    library_specs = splits["source_train"]
    if not library_specs:
        # A chip collection normally defines families rather than source tasks.
        # One representative per family is enough to induce the library.
        seen: dict[str, TaskSpec] = {}
        for name in SPLITS:
            for task in splits[name]:
                seen.setdefault(task.family, task)
        library_specs = list(seen.values())
    if not library_specs:
        raise ManifestError("no entries to induce a skill library from")
    return splits, scenes, build_library(library_specs)


def write_manifest(path: str | Path, payload: dict) -> Path:
    """Write a manifest, used by the chip-preparation helpers and the tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
