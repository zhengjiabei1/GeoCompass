"""Tests for the dependency-free GeoTIFF reader and test-only writer."""
from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from geocompass.io import (
    GeoTIFFError,
    read_geotiff,
    read_geotiff_scene,
    write_geotiff,
)

# Real-data check (optional).  Set GEOTIFF_SAMPLE to a local GeoTIFF path to
# enable it.  Public samples used while developing this reader:
#
#   curl -L -o /tmp/RGB.byte.tif \
#     https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif
#   curl -L -o /tmp/rgb_deflate.tif \
#     https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/rgb_deflate.tif
#   curl -L -o /tmp/byte_bigtiff_strip5lines.tif \
#     https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/gcore/data/byte_bigtiff_strip5lines.tif
#
# The rasterio files are three-band uint8 GeoTIFFs distributed with rasterio's
# test data under the BSD-3-Clause licence.  The GDAL file is a uint8 BigTIFF
# distributed with GDAL's autotest data under the MIT licence.  No binary is
# stored in this repository.
REAL_SAMPLE = os.environ.get("GEOTIFF_SAMPLE")

_TYPE_CHARS = {3: "H", 4: "I", 11: "f", 12: "d"}


def _pack_values(endian: str, ttype: int, values) -> bytes:
    if ttype == 2:
        return values.encode("ascii") + b"\x00"
    return struct.pack(endian + str(len(values)) + _TYPE_CHARS[ttype], *values)


def build_tiff(entries, payload: bytes, *, byteorder: str = "<",
               bigtiff: bool = False) -> bytes:
    """Assemble a classic or BigTIFF file from ``(tag, type, values)`` entries.

    ``payload`` is placed directly after the header; the IFD follows it and
    tagged values that do not fit inline are stored after the IFD.
    """
    header_size = 16 if bigtiff else 8
    field = 8 if bigtiff else 4
    entry_size = 20 if bigtiff else 12
    ifd_offset = header_size + len(payload)
    ifd_size = (8 if bigtiff else 2) + entry_size * len(entries) + field
    extra_base = ifd_offset + ifd_size
    extra = bytearray()
    body = bytearray()
    for tag, ttype, values in entries:
        raw = _pack_values(byteorder, ttype, values)
        count = len(raw) if ttype == 2 else len(values)
        if len(raw) <= field:
            value_field = raw.ljust(field, b"\x00")
        else:
            value_field = struct.pack(
                byteorder + ("Q" if bigtiff else "I"), extra_base + len(extra)
            )
            extra.extend(raw)
            if len(extra) % 2:
                extra.append(0)
        body.extend(struct.pack(byteorder + "HH", tag, ttype))
        body.extend(struct.pack(byteorder + ("Q" if bigtiff else "I"), count))
        body.extend(value_field)
    marker = b"II" if byteorder == "<" else b"MM"
    if bigtiff:
        header = marker + struct.pack(byteorder + "HHH", 43, 8, 0)
        header += struct.pack(byteorder + "Q", ifd_offset)
        ifd = struct.pack(byteorder + "Q", len(entries)) + bytes(body)
        ifd += struct.pack(byteorder + "Q", 0)
    else:
        header = marker + struct.pack(byteorder + "HI", 42, ifd_offset)
        ifd = struct.pack(byteorder + "H", len(entries)) + bytes(body)
        ifd += struct.pack(byteorder + "I", 0)
    return header + payload + ifd + bytes(extra)


def base_entries(width: int, height: int, *, bits: int = 16, samples: int = 1,
                 sample_format: int = 1, compression: int = 1,
                 predictor: int | None = None, planar: int = 1):
    entries = [
        (256, 4, [width]),
        (257, 4, [height]),
        (258, 3, [bits] * samples),
        (259, 3, [compression]),
        (262, 3, [2 if samples >= 3 else 1]),
        (277, 3, [samples]),
        (284, 3, [planar]),
        (339, 3, [sample_format] * samples),
    ]
    if predictor is not None:
        entries.append((317, 3, [predictor]))
    return entries


class RasterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def path(self, name: str) -> Path:
        return self.root / name


class TestRoundTrip(RasterTestCase):
    def test_uint16_single_band(self) -> None:
        values = [0, 1, 2, 300, 65535, 42]
        write_geotiff(self.path("u16.tif"), [values], width=3, height=2)
        raster = read_geotiff(self.path("u16.tif"))
        self.assertEqual((raster.width, raster.height), (3, 2))
        self.assertEqual(raster.bands, [[float(value) for value in values]])
        self.assertEqual(raster.band_names, ["band_1"])
        self.assertIsNone(raster.nodata)

    def test_float32_single_band(self) -> None:
        values = [0.5, -1.25, 2.0, 8.0, -0.125, 100.5]
        write_geotiff(
            self.path("f32.tif"), [values], width=2, height=3,
            sample_format="float32",
        )
        raster = read_geotiff(self.path("f32.tif"))
        for expected, actual in zip(values, raster.bands[0]):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_three_band_chunky(self) -> None:
        bands = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [100, 200, 300, 400],
        ]
        write_geotiff(self.path("rgb.tif"), bands, width=2, height=2)
        raster = read_geotiff(self.path("rgb.tif"))
        self.assertEqual(raster.n_bands, 3)
        self.assertEqual(raster.bands, [[1.0, 2.0, 3.0, 4.0],
                                        [10.0, 20.0, 30.0, 40.0],
                                        [100.0, 200.0, 300.0, 400.0]])

    def test_multi_strip(self) -> None:
        values = [float(index) for index in range(15)]
        write_geotiff(self.path("strips.tif"), [values], width=3, height=5,
                      rows_per_strip=2)
        raster = read_geotiff(self.path("strips.tif"))
        self.assertEqual(raster.bands[0], values)
        self.assertEqual(raster.width, 3)
        self.assertEqual(raster.height, 5)

    def test_big_endian_writer(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        write_geotiff(self.path("big.tif"), [values], width=3, height=2,
                      byteorder=">")
        self.assertEqual(self.path("big.tif").read_bytes()[:2], b"MM")
        raster = read_geotiff(self.path("big.tif"))
        self.assertEqual(raster.bands[0], values)

    def test_geo_tags_round_trip(self) -> None:
        geo = {
            "ModelPixelScaleTag": [10.0, 10.0, 0.0],
            "ModelTiepointTag": [0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0],
        }
        write_geotiff(self.path("geo.tif"), [[1, 2, 3, 4]], width=2, height=2,
                      geo_tags=geo)
        raster = read_geotiff(self.path("geo.tif"))
        self.assertEqual(raster.geo_tags["ModelPixelScaleTag"], [10.0, 10.0, 0.0])
        self.assertEqual(
            raster.geo_tags["ModelTiepointTag"],
            [0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0],
        )


class TestSpecialLayouts(RasterTestCase):
    def test_deflate_compressed_strip(self) -> None:
        values = [1, 2, 3, 4, 5, 6]
        raw = struct.pack("<6H", *values)
        payload = zlib.compress(raw)
        entries = base_entries(3, 2, compression=8)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [len(payload)])]
        self.path("deflate.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("deflate.tif"))
        self.assertEqual(raster.bands[0], [float(value) for value in values])

    def test_predictor_2_horizontal_differencing(self) -> None:
        rows = [[10, 12, 15, 20], [5, 5, 5, 5]]
        encoded = []
        for row in rows:
            encoded.append(row[0])
            for previous, value in zip(row, row[1:]):
                encoded.append((value - previous) % 65536)
        payload = struct.pack("<8H", *encoded)
        entries = base_entries(4, 2, predictor=2)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [len(payload)])]
        self.path("predictor.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("predictor.tif"))
        self.assertEqual(raster.bands[0], [10.0, 12.0, 15.0, 20.0,
                                           5.0, 5.0, 5.0, 5.0])

    def test_deflate_with_predictor_2(self) -> None:
        rows = [[100, 90, 80], [70, 60, 50]]
        encoded = []
        for row in rows:
            encoded.append(row[0])
            for previous, value in zip(row, row[1:]):
                encoded.append((value - previous) % 65536)
        payload = zlib.compress(struct.pack("<6H", *encoded))
        entries = base_entries(3, 2, compression=8, predictor=2)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [len(payload)])]
        self.path("deflate_predictor.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("deflate_predictor.tif"))
        self.assertEqual(raster.bands[0], [100.0, 90.0, 80.0, 70.0, 60.0, 50.0])

    def test_bigtiff_header(self) -> None:
        values = [7, 8, 9, 10]
        payload = struct.pack("<4H", *values)
        entries = base_entries(2, 2)
        entries += [(273, 4, [16]), (278, 4, [2]), (279, 4, [len(payload)])]
        self.path("bigtiff.tif").write_bytes(
            build_tiff(entries, payload, bigtiff=True)
        )
        self.assertEqual(self.path("bigtiff.tif").read_bytes()[2:4], b"+\x00")
        raster = read_geotiff(self.path("bigtiff.tif"))
        self.assertEqual(raster.bands[0], [7.0, 8.0, 9.0, 10.0])

    def test_planar_configuration(self) -> None:
        band0 = [1, 2, 3, 4]
        band1 = [10, 20, 30, 40]
        payload = struct.pack("<4H", *band0) + struct.pack("<4H", *band1)
        entries = base_entries(2, 2, samples=2, planar=2)
        entries += [(273, 4, [8, 16]), (278, 4, [2]), (279, 4, [8, 8])]
        self.path("planar.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("planar.tif"))
        self.assertEqual(raster.bands, [[1.0, 2.0, 3.0, 4.0],
                                        [10.0, 20.0, 30.0, 40.0]])

    def test_tiled_layout(self) -> None:
        values = list(range(16))
        tiles = [
            [0, 1, 4, 5], [2, 3, 6, 7],
            [8, 9, 12, 13], [10, 11, 14, 15],
        ]
        payload = b"".join(bytes(tile) for tile in tiles)
        entries = base_entries(4, 4, bits=8)
        entries += [
            (322, 4, [2]),
            (323, 4, [2]),
            (324, 4, [8, 12, 16, 20]),
            (325, 4, [4, 4, 4, 4]),
        ]
        self.path("tiled.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("tiled.tif"))
        self.assertEqual(raster.bands[0], [float(value) for value in values])

    def test_signed_int16(self) -> None:
        values = [-1, 0, 100, -200]
        payload = struct.pack("<4h", *values)
        entries = base_entries(2, 2, sample_format=2)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [len(payload)])]
        self.path("int16.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("int16.tif"))
        self.assertEqual(raster.bands[0], [float(value) for value in values])

    def test_gdal_nodata_tag(self) -> None:
        payload = struct.pack("<4H", 1, 2, 3, 4)
        entries = base_entries(2, 2)
        entries += [
            (273, 4, [8]),
            (278, 4, [2]),
            (279, 4, [len(payload)]),
            (42113, 2, "-9999.0"),
        ]
        self.path("nodata.tif").write_bytes(build_tiff(entries, payload))
        raster = read_geotiff(self.path("nodata.tif"))
        self.assertEqual(raster.nodata, -9999.0)


class TestErrors(RasterTestCase):
    def test_unsupported_compression(self) -> None:
        entries = base_entries(2, 2, compression=7)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [4])]
        self.path("jpeg.tif").write_bytes(
            build_tiff(entries, struct.pack("<2H", 1, 2))
        )
        with self.assertRaises(GeoTIFFError) as context:
            read_geotiff(self.path("jpeg.tif"))
        self.assertIn("compression", str(context.exception))

    def test_unknown_predictor(self) -> None:
        entries = base_entries(2, 2, predictor=3)
        entries += [(273, 4, [8]), (278, 4, [2]), (279, 4, [8])]
        self.path("predictor3.tif").write_bytes(
            build_tiff(entries, struct.pack("<4H", 1, 2, 3, 4))
        )
        with self.assertRaises(GeoTIFFError):
            read_geotiff(self.path("predictor3.tif"))

    def test_not_a_tiff(self) -> None:
        self.path("bad.bin").write_bytes(b"not a tiff at all")
        with self.assertRaises(GeoTIFFError):
            read_geotiff(self.path("bad.bin"))


class TestSceneHelper(RasterTestCase):
    def test_single_date_with_labels(self) -> None:
        write_geotiff(self.path("red.tif"), [[1, 2, 3, 4]], width=2, height=2)
        write_geotiff(self.path("nir.tif"), [[10, 20, 30, 40]], width=2, height=2)
        write_geotiff(self.path("labels.tif"), [[0, 1, 1, 5]], width=2, height=2)
        scene = read_geotiff_scene(
            [self.path("red.tif"), self.path("nir.tif")],
            band_names=["red", "nir"],
            labels_path=self.path("labels.tif"),
        )
        self.assertEqual(sorted(scene.pre), ["nir", "red"])
        self.assertEqual(scene.pre["red"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(scene.pre["nir"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(scene.labels, [0, 1, 1, 5])
        self.assertIsNone(scene.post)
        self.assertEqual((scene.width, scene.height), (2, 2))

    def test_bitemporal_pair(self) -> None:
        write_geotiff(self.path("pre.tif"), [[1, 2, 3, 4]], width=2, height=2)
        write_geotiff(self.path("post.tif"), [[5, 6, 7, 8]], width=2, height=2)
        scene = read_geotiff_scene(
            self.path("pre.tif"),
            band_names=["red"],
            post_paths=self.path("post.tif"),
        )
        self.assertEqual(scene.pre["red"], [1.0, 2.0, 3.0, 4.0])
        self.assertIsNotNone(scene.post)
        self.assertEqual(scene.post["red"], [5.0, 6.0, 7.0, 8.0])

    def test_multiband_file_uses_supplied_names(self) -> None:
        bands = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        write_geotiff(self.path("rgb.tif"), bands, width=2, height=2)
        scene = read_geotiff_scene(
            self.path("rgb.tif"), band_names=["red", "green", "blue"],
        )
        self.assertEqual(list(scene.pre), ["red", "green", "blue"])
        self.assertEqual(scene.pre["green"], [5.0, 6.0, 7.0, 8.0])

    def test_label_shape_mismatch(self) -> None:
        write_geotiff(self.path("two.tif"), [[1, 2, 3, 4]], width=2, height=2)
        write_geotiff(self.path("nine.tif"), [list(range(9))], width=3, height=3)
        with self.assertRaises(ValueError):
            read_geotiff_scene(
                self.path("two.tif"), band_names=["red"],
                labels_path=self.path("nine.tif"),
            )

    def test_resample_to_square_chip(self) -> None:
        values = [float(index) for index in range(16)]
        write_geotiff(self.path("four.tif"), [values], width=4, height=4)
        write_geotiff(self.path("labels4.tif"), [list(range(16))], width=4, height=4)
        scene = read_geotiff_scene(
            self.path("four.tif"), band_names=["red"],
            labels_path=self.path("labels4.tif"), size=2,
        )
        self.assertEqual(len(scene.pre["red"]), 4)
        self.assertEqual(scene.pre["red"], [0.0, 2.0, 8.0, 10.0])
        self.assertEqual(scene.labels, [0, 2, 8, 10])
        self.assertEqual((scene.width, scene.height), (2, 2))


@unittest.skipUnless(REAL_SAMPLE, "set GEOTIFF_SAMPLE to a local GeoTIFF path")
class TestRealData(unittest.TestCase):
    def test_read_public_sample(self) -> None:
        raster = read_geotiff(REAL_SAMPLE)
        self.assertGreater(raster.width, 0)
        self.assertGreater(raster.height, 0)
        self.assertEqual(raster.n_bands, len(raster.bands))
        for band in raster.bands:
            self.assertEqual(len(band), raster.width * raster.height)
        self.assertEqual(list(raster.geo_tags), [
            name for name in ("ModelPixelScaleTag", "ModelTiepointTag",
                              "ModelTransformationTag", "GeoKeyDirectoryTag")
            if name in raster.geo_tags
        ])


if __name__ == "__main__":
    unittest.main()
