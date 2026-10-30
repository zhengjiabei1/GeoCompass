"""Tests for the chip-conversion helpers used by the preparation script."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from geocompass.io import read_geotiff, write_geotiff
from geocompass.prepare import (
    Chip,
    PrepareError,
    SEN1FLOODS11_BANDS,
    convert_image,
    convert_labels,
    manifest_payload,
    write_manifest,
)

SIZE = 4


def _thirteen_band_source(path: Path) -> None:
    bands = [[float(100 * (index + 1) + row) for row in range(SIZE * SIZE)]
             for index in range(13)]
    write_geotiff(path, bands, width=SIZE, height=SIZE, sample_format="uint16")


class ConvertImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        _thirteen_band_source(self.root / "chip_S2Hand.tif")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_selects_six_bands_and_scales(self) -> None:
        out = convert_image(self.root / "chip_S2Hand.tif", self.root / "out.tif",
                            SEN1FLOODS11_BANDS, scale=1e-4)
        raster = read_geotiff(out)
        self.assertEqual(len(raster.bands), 6)
        # Band 1 of the selection is source band index 1, whose first value is 200.
        self.assertAlmostEqual(raster.bands[0][0], 0.02, places=6)
        # Band 6 is source index 12, first value 1300.
        self.assertAlmostEqual(raster.bands[5][0], 0.13, places=6)

    def test_wrong_band_count(self) -> None:
        with self.assertRaises(PrepareError):
            convert_image(self.root / "chip_S2Hand.tif", self.root / "out.tif", (1, 2))

    def test_band_index_out_of_range(self) -> None:
        with self.assertRaises(PrepareError):
            convert_image(self.root / "chip_S2Hand.tif", self.root / "out.tif",
                          (1, 2, 3, 7, 11, 40))


class ConvertLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        values = [-1.0, 0.0, 1.0, 1.0]
        write_geotiff(self.root / "label.tif", [values], width=2, height=2,
                      sample_format="int16")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_nodata_is_remapped(self) -> None:
        out = convert_labels(self.root / "label.tif", self.root / "out.tif")
        raster = read_geotiff(out)
        self.assertEqual(raster.bands[0][0], 65535)
        self.assertEqual(raster.bands[0][1:], [0, 1, 1])

    def test_multi_band_label_is_rejected(self) -> None:
        write_geotiff(self.root / "two.tif", [[1.0, 0.0], [0.0, 1.0]], width=2, height=1,
                      sample_format="uint16")
        with self.assertRaises(PrepareError):
            convert_labels(self.root / "two.tif", self.root / "out.tif")


class ManifestTests(unittest.TestCase):
    def test_payload_round_trip(self) -> None:
        chips = [Chip("a", "a.tif", "a_label.tif", "flood", "target_test")]
        payload = manifest_payload(chips, name="fixture", size=32)
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(payload, Path(directory) / "manifest.json")
            self.assertEqual(json.loads(path.read_text())["entries"][0]["id"], "a")


if __name__ == "__main__":
    unittest.main()
