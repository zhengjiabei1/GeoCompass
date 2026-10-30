"""Tests for the real-data scene adapter."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from geocompass.io import write_geotiff
from geocompass.ops import execute_skill
from geocompass.realdata import MissingBandError, SceneSpec, load_scene, scene_from_rasters
from geocompass.tasks import generic_pipeline
from geocompass.types import BANDS, Scene, TaskSpec


def _chip(size: int = 8) -> list[list[float]]:
    """Six bands with a water block in the left half."""
    bands = []
    for band_index, name in enumerate(BANDS):
        values = []
        for row in range(size):
            for column in range(size):
                water = column < size // 2
                if name == "nir":
                    value = 800 if water else 4000
                elif name in {"swir1", "swir2"}:
                    value = 300 if water else 5000
                else:
                    value = 900 if water else 1400
                values.append(float(value + band_index * 3 + row))
        bands.append(values)
    return bands


def _labels(size: int = 8) -> list[float]:
    return [1.0 if column < size // 2 else 0.0
            for _ in range(size) for column in range(size)]


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.size = 8
        write_geotiff(self.root / "bands.tif", _chip(self.size), width=self.size,
                      height=self.size, sample_format="float32")
        write_geotiff(self.root / "labels.tif", [_labels(self.size)], width=self.size,
                      height=self.size, sample_format="uint16")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_scene_fields(self) -> None:
        scene = load_scene(self.root / "bands.tif", band_names=BANDS,
                           labels_path=self.root / "labels.tif")
        self.assertIsInstance(scene, Scene)
        self.assertEqual(set(scene.pre), set(BANDS))
        self.assertEqual(len(scene.pre["nir"]), self.size * self.size)
        self.assertEqual(scene.truth_kind, "binary")
        self.assertTrue(all(cell == 1 for cell in scene.valid))
        self.assertEqual(len(scene.samples), len(scene.sample_labels))
        self.assertGreater(len(scene.samples), 0)

    def test_scene_is_executable(self) -> None:
        scene = load_scene(self.root / "bands.tif", band_names=BANDS,
                           labels_path=self.root / "labels.tif")
        task = TaskSpec(task_id="real-1", family="water", split="test", seed=0,
                        size=self.size)
        context = execute_skill(generic_pipeline(task), task, scene)
        self.assertEqual(len(context.map), self.size * self.size)

    def test_missing_band_is_an_error(self) -> None:
        import geocompass.io as io_module

        rasters = io_module.read_geotiff_scene(self.root / "bands.tif", band_names=BANDS)
        partial = io_module.RasterScene(
            pre={key: value for key, value in rasters.pre.items() if key != "nir"},
            post=None, labels=None, width=self.size, height=self.size)
        with self.assertRaises(MissingBandError):
            scene_from_rasters(partial)

    def test_scale_and_offset(self) -> None:
        import geocompass.io as io_module

        rasters = io_module.read_geotiff_scene(self.root / "bands.tif",
                                               band_names=BANDS)
        scene = scene_from_rasters(rasters, SceneSpec(scale=1e-4, offset=0.0), BANDS)
        self.assertLess(max(scene.pre["nir"]), 1.0)

    def test_resampling_to_target_size(self) -> None:
        scene = load_scene(self.root / "bands.tif", band_names=BANDS,
                           labels_path=self.root / "labels.tif", size=4)
        self.assertEqual(scene.size, 4)
        self.assertEqual(len(scene.pre["nir"]), 16)
        self.assertEqual(len(scene.truth), 16)

    def test_nodata_becomes_invalid(self) -> None:
        import geocompass.io as io_module

        rasters = io_module.read_geotiff_scene(self.root / "bands.tif",
                                               band_names=BANDS)
        rasters.nodata = rasters.pre["nir"][0]
        scene = scene_from_rasters(rasters, SceneSpec(), BANDS)
        self.assertEqual(scene.valid[0], 0)


if __name__ == "__main__":
    unittest.main()
