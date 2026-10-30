"""Tests for the chip-manifest loader."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from geocompass.dataset import ManifestError, load_dataset, parse_entries, read_manifest
from geocompass.io import write_geotiff
from geocompass.raster import generate_scene
from geocompass.tasks import build_task_specs
from geocompass.types import BANDS

SIZE = 16


def _chip(directory: Path, name: str, spec) -> tuple[str, str]:
    scene = generate_scene(spec)
    bands = [scene.pre[band] for band in BANDS]
    image = directory / f"{name}.tif"
    write_geotiff(image, bands, width=scene.size, height=scene.size, sample_format="float32")
    label = directory / f"{name}_label.tif"
    write_geotiff(label, [scene.truth], width=scene.size, height=scene.size,
                  sample_format="uint16")
    return image.name, label.name


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.specs = build_task_specs(0, {"source_train": 1, "target_test": 1})
        self.payload = {
            "name": "fixture",
            "band_names": list(BANDS),
            "scale": 1.0,
            "size": SIZE,
            "entries": [],
        }
        for split, tasks in self.specs.items():
            for task in tasks:
                image, label = _chip(self.root, task.task_id, task)
                self.payload["entries"].append({
                    "id": task.task_id,
                    "family": task.family,
                    "split": split,
                    "pre": image,
                    "labels": label,
                })
        (self.root / "manifest.json").write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_manifest_is_read_from_directory(self) -> None:
        manifest = read_manifest(self.root)
        self.assertEqual(manifest["name"], "fixture")
        self.assertEqual(len(manifest["entries"]), len(self.specs["source_train"])
                         + len(self.specs["target_test"]))

    def test_entries_resolve_paths(self) -> None:
        entries = parse_entries(read_manifest(self.root))
        self.assertTrue(all(Path(entry.pre[0]).is_absolute() for entry in entries))
        self.assertTrue(all(entry.labels for entry in entries))

    def test_dataset_builds_splits_and_library(self) -> None:
        splits, scenes, library = load_dataset(self.root)
        self.assertEqual(len(splits["source_train"]), len(self.specs["source_train"]))
        self.assertEqual(len(splits["target_test"]), len(self.specs["target_test"]))
        self.assertTrue(scenes)
        sample = next(iter(scenes.values()))
        self.assertEqual(set(sample.pre), set(BANDS))
        self.assertEqual(len(sample.truth), sample.size * sample.size)
        self.assertTrue(library)

    def test_unknown_family_is_rejected(self) -> None:
        payload = dict(self.payload)
        payload["entries"] = [dict(payload["entries"][0], family="lava")]
        (self.root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ManifestError):
            read_manifest(self.root) and parse_entries(read_manifest(self.root))

    def test_missing_file_is_rejected(self) -> None:
        payload = dict(self.payload)
        payload["entries"] = [dict(payload["entries"][0], pre="absent.tif")]
        (self.root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ManifestError):
            parse_entries(read_manifest(self.root))

    def test_duplicate_ids_are_rejected(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["entries"].append(dict(payload["entries"][0]))
        (self.root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ManifestError):
            parse_entries(read_manifest(self.root))

    def test_missing_manifest(self) -> None:
        with self.assertRaises(ManifestError):
            read_manifest(self.root / "nope.json")


if __name__ == "__main__":
    unittest.main()
