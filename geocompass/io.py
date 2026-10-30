"""Dependency-free GeoTIFF reading for Earth-observation chips.

The module implements the subset of the TIFF/GeoTIFF specification that is
needed to turn small scenes into plain Python lists:

* classic TIFF (``II`` and ``MM``) and BigTIFF headers;
* striped and tiled layouts;
* uncompressed and Deflate (zlib) segments;
* the horizontal-differencing predictor (Predictor = 2);
* 8/16/32/64-bit unsigned and signed integers, 16/32/64-bit floats;
* chunky (PlanarConfiguration = 1) and planar (PlanarConfiguration = 2)
  sample storage;
* the model pixel-scale, model tiepoint and model transformation tags.

Only the first image file directory of a file is read.  LZW, JPEG, WebP and
other compressed encodings, per-sample bit depths that differ inside one file,
and palettes expanded to RGB are not supported and raise :class:`GeoTIFFError`.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "GeoRaster",
    "GeoTIFFError",
    "RasterScene",
    "read_geotiff",
    "read_geotiff_scene",
    "write_geotiff",
]

# TIFF field tags used by the reader.
_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_BITS_PER_SAMPLE = 258
_COMPRESSION = 259
_PHOTOMETRIC = 262
_STRIP_OFFSETS = 273
_SAMPLES_PER_PIXEL = 277
_ROWS_PER_STRIP = 278
_STRIP_BYTE_COUNTS = 279
_PLANAR_CONFIGURATION = 284
_PREDICTOR = 317
_TILE_WIDTH = 322
_TILE_LENGTH = 323
_TILE_OFFSETS = 324
_TILE_BYTE_COUNTS = 325
_SAMPLE_FORMAT = 339
_MODEL_PIXEL_SCALE = 33550
_MODEL_TIEPOINT = 33922
_MODEL_TRANSFORMATION = 34264
_GEO_KEY_DIRECTORY = 34735
_GDAL_NODATA = 42113

_TYPE_SIZE = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD
    16: 8,  # LONG8
    17: 8,  # SLONG8
    18: 8,  # IFD8
}
_TYPE_CHAR = {
    1: "B",
    3: "H",
    4: "I",
    6: "b",
    8: "h",
    9: "i",
    11: "f",
    12: "d",
    13: "I",
    16: "Q",
    17: "q",
    18: "Q",
}
_UNSIGNED_CHAR = {8: "B", 16: "H", 32: "I", 64: "Q"}
_SIGNED_CHAR = {8: "b", 16: "h", 32: "i", 64: "q"}
_FLOAT_CHAR = {16: "e", 32: "f", 64: "d"}

_GEO_TAG_ALIASES = {
    "ModelPixelScaleTag": _MODEL_PIXEL_SCALE,
    "ModelTiepointTag": _MODEL_TIEPOINT,
    "ModelTransformationTag": _MODEL_TRANSFORMATION,
    "GeoKeyDirectoryTag": _GEO_KEY_DIRECTORY,
}


class GeoTIFFError(ValueError):
    """Raised for malformed or unsupported TIFF/GeoTIFF content."""


@dataclass
class GeoRaster:
    """Decoded raster content plus the geo-referencing tags we understand."""

    width: int
    height: int
    bands: list[list[float]]
    band_names: list[str]
    nodata: float | None
    geo_tags: dict[str, Any]

    @property
    def n_bands(self) -> int:
        return len(self.bands)


@dataclass
class RasterScene:
    """Raw arrays for one or two dates, before the project builds a ``Scene``."""

    pre: dict[str, list[float]]
    post: dict[str, list[float]] | None
    labels: list[int] | None
    width: int
    height: int
    nodata: float | None = None
    geo_tags: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level TIFF parsing
# ---------------------------------------------------------------------------


def _parse_header(data: bytes, target: Path) -> tuple[str, bool, int]:
    """Return ``(endianness, is_bigtiff, first_ifd_offset)``."""
    if len(data) < 8:
        raise GeoTIFFError(f"{target}: file is too small to be a TIFF")
    marker = data[:2]
    if marker == b"II":
        endian = "<"
    elif marker == b"MM":
        endian = ">"
    else:
        raise GeoTIFFError(f"{target}: not a TIFF file (byte-order marker {marker!r})")
    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic == 42:
        return endian, False, struct.unpack(endian + "I", data[4:8])[0]
    if magic == 43:
        if len(data) < 16:
            raise GeoTIFFError(f"{target}: truncated BigTIFF header")
        offset_size, reserved = struct.unpack(endian + "HH", data[4:8])
        if offset_size != 8 or reserved != 0:
            raise GeoTIFFError(f"{target}: invalid BigTIFF header")
        return endian, True, struct.unpack(endian + "Q", data[8:16])[0]
    raise GeoTIFFError(f"{target}: unsupported TIFF version {magic}")


def _decode_value(endian: str, ttype: int, count: int, raw: bytes) -> Any:
    if ttype == 2:
        return raw.split(b"\x00", 1)[0].decode("ascii", "replace")
    if ttype == 7:
        return raw
    if count == 0:
        return []
    if ttype in (5, 10):
        char = "I" if ttype == 5 else "i"
        pairs = struct.unpack(endian + char * (2 * count), raw)
        values = []
        for index in range(0, 2 * count, 2):
            denominator = pairs[index + 1]
            values.append(pairs[index] / denominator if denominator else 0.0)
        return values
    char = _TYPE_CHAR.get(ttype)
    if char is None:
        return raw
    return list(struct.unpack(endian + str(count) + char, raw))


def _read_ifd(data: bytes, target: Path, endian: str, big: bool,
              offset: int) -> tuple[dict[int, Any], int]:
    field_size = 8 if big else 4
    entry_size = 20 if big else 12
    count_fmt = "Q" if big else "H"
    count_size = 8 if big else 2
    if offset <= 0 or offset + count_size > len(data):
        raise GeoTIFFError(f"{target}: first IFD offset {offset} is outside the file")
    count = struct.unpack(endian + count_fmt, data[offset:offset + count_size])[0]
    entries_start = offset + count_size
    if entries_start + count * entry_size > len(data):
        raise GeoTIFFError(f"{target}: IFD with {count} entries is truncated")
    tags: dict[int, Any] = {}
    for index in range(count):
        base = entries_start + index * entry_size
        tag, ttype = struct.unpack(endian + "HH", data[base:base + 4])
        count_value = struct.unpack(
            endian + ("Q" if big else "I"), data[base + 4:base + 4 + field_size]
        )[0]
        size = _TYPE_SIZE.get(ttype)
        if size is None:
            continue
        value_pos = base + 4 + field_size
        total = size * count_value
        if total <= field_size:
            raw = data[value_pos:value_pos + total]
        else:
            pointer = struct.unpack(
                endian + ("Q" if big else "I"), data[value_pos:value_pos + field_size]
            )[0]
            if pointer + total > len(data):
                raise GeoTIFFError(f"{target}: value of tag {tag} lies outside the file")
            raw = data[pointer:pointer + total]
        if len(raw) < total:
            raise GeoTIFFError(f"{target}: tag {tag} is truncated")
        tags[tag] = _decode_value(endian, ttype, count_value, raw)
    next_offset = entries_start + count * entry_size
    next_ifd = struct.unpack(
        endian + ("Q" if big else "I"), data[next_offset:next_offset + field_size]
    )[0]
    return tags, next_ifd


def _tag_value(tags: dict[int, Any], tag: int, default: Any = None) -> Any:
    value = tags.get(tag)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _tag_list(tags: dict[int, Any], tag: int, default: list[Any] | None = None) -> list[Any]:
    value = tags.get(tag)
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _uniform(values: list[Any], samples: int, name: str, target: Path) -> int:
    if not values:
        raise GeoTIFFError(f"{target}: empty {name} tag")
    first = int(values[0])
    if len(values) not in (1, samples):
        raise GeoTIFFError(
            f"{target}: {name} has {len(values)} entries for {samples} samples"
        )
    if any(int(value) != first for value in values):
        raise GeoTIFFError(
            f"{target}: per-sample {name} values differ, which is unsupported"
        )
    return first


def _unpack(raw: bytes, count: int, dtype: str, endian: str, target: Path) -> list[Any]:
    need = count * struct.calcsize(dtype)
    if len(raw) < need:
        raise GeoTIFFError(
            f"{target}: segment holds {len(raw)} bytes but {need} are required"
        )
    try:
        return list(struct.unpack(endian + str(count) + dtype, raw[:need]))
    except struct.error as exc:  # pragma: no cover - defensive
        raise GeoTIFFError(f"{target}: cannot unpack segment: {exc}") from exc


def _decode_segment(data: bytes, target: Path, offset: int, byte_count: int,
                    compression: int, rows: int, cols: int, components: int,
                    dtype: str, bits: int, predictor: int, signed: bool,
                    endian: str) -> list[float]:
    """Decode one strip or tile into row-major floats."""
    if offset < 0 or byte_count < 0 or offset + byte_count > len(data):
        raise GeoTIFFError(f"{target}: segment points outside the file")
    raw = data[offset:offset + byte_count]
    if compression != 1:
        try:
            raw = zlib.decompress(raw)
        except zlib.error as exc:
            raise GeoTIFFError(f"{target}: cannot inflate Deflate segment: {exc}") from exc
    read_dtype = _UNSIGNED_CHAR[bits] if predictor == 2 else dtype
    values = _unpack(raw, rows * cols * components, read_dtype, endian, target)
    if predictor == 2:
        mask = (1 << bits) - 1
        row_span = cols * components
        for row in range(rows):
            start = row * row_span
            for index in range(start + components, start + row_span):
                values[index] = (values[index] + values[index - components]) & mask
        if signed:
            half = 1 << (bits - 1)
            span = 1 << bits
            values = [value - span if value >= half else value for value in values]
    return [float(value) for value in values]


def _strip_byte_counts(tags: dict[int, Any], strips: int, rows_per_strip: int,
                       height: int, width: int, components: int,
                       bits: int) -> list[int]:
    counts = [int(value) for value in _tag_list(tags, _STRIP_BYTE_COUNTS, [])]
    bytes_per_sample = bits // 8
    while len(counts) < strips:
        strip = len(counts) % strips
        rows = min(rows_per_strip, height - strip * rows_per_strip)
        counts.append(rows * width * components * bytes_per_sample)
    return counts


def _read_strips(data: bytes, target: Path, tags: dict[int, Any],
                 width: int, height: int, samples: int, dtype: str, bits: int,
                 predictor: int, signed: bool, planar: int, compression: int,
                 endian: str) -> list[list[float]]:
    rows_per_strip = int(_tag_value(tags, _ROWS_PER_STRIP, height) or height)
    rows_per_strip = max(1, min(rows_per_strip, height))
    strips = (height + rows_per_strip - 1) // rows_per_strip
    offsets = [int(value) for value in _tag_list(tags, _STRIP_OFFSETS, [])]
    required = strips * samples if planar == 2 else strips
    if len(offsets) < required:
        raise GeoTIFFError(
            f"{target}: {len(offsets)} strip offsets for {required} strips"
        )
    counts = _strip_byte_counts(
        tags, required, rows_per_strip, height, width,
        samples if planar == 1 else 1, bits,
    )
    bands: list[list[float]] = [[] for _ in range(samples)]
    if planar == 1:
        for strip in range(strips):
            row0 = strip * rows_per_strip
            rows = min(rows_per_strip, height - row0)
            values = _decode_segment(
                data, target, offsets[strip], counts[strip], compression,
                rows, width, samples, dtype, bits, predictor, signed, endian,
            )
            for band in range(samples):
                bands[band].extend(values[band::samples])
    else:
        for band in range(samples):
            base = band * strips
            for strip in range(strips):
                row0 = strip * rows_per_strip
                rows = min(rows_per_strip, height - row0)
                values = _decode_segment(
                    data, target, offsets[base + strip], counts[base + strip],
                    compression, rows, width, 1, dtype, bits, predictor,
                    signed, endian,
                )
                bands[band].extend(values)
    return bands


def _read_tiles(data: bytes, target: Path, tags: dict[int, Any],
                width: int, height: int, samples: int, dtype: str, bits: int,
                predictor: int, signed: bool, planar: int, compression: int,
                endian: str) -> list[list[float]]:
    tile_width = int(_tag_value(tags, _TILE_WIDTH, 0))
    tile_height = int(_tag_value(tags, _TILE_LENGTH, 0))
    if tile_width <= 0 or tile_height <= 0:
        raise GeoTIFFError(f"{target}: invalid tile geometry")
    across = (width + tile_width - 1) // tile_width
    down = (height + tile_height - 1) // tile_height
    per_plane = across * down
    planes = samples if planar == 2 else 1
    offsets = [int(value) for value in _tag_list(tags, _TILE_OFFSETS, [])]
    required = per_plane * planes
    if len(offsets) < required:
        raise GeoTIFFError(f"{target}: {len(offsets)} tile offsets for {required} tiles")
    raw_counts = _tag_list(tags, _TILE_BYTE_COUNTS, [])
    counts = [int(value) for value in raw_counts]
    bytes_per_sample = bits // 8
    tile_components = samples if planar == 1 else 1
    while len(counts) < required:
        counts.append(tile_width * tile_height * tile_components * bytes_per_sample)

    result = [[0.0] * (width * height) for _ in range(samples)]
    for plane in range(planes):
        for tile_y in range(down):
            for tile_x in range(across):
                index = plane * per_plane + tile_y * across + tile_x
                components = 1 if planar == 2 else samples
                values = _decode_segment(
                    data, target, offsets[index], counts[index], compression,
                    tile_height, tile_width, components, dtype, bits, predictor,
                    signed, endian,
                )
                rows = min(tile_height, height - tile_y * tile_height)
                cols = min(tile_width, width - tile_x * tile_width)
                band = plane if planar == 2 else 0
                for row in range(rows):
                    destination = (tile_y * tile_height + row) * width + tile_x * tile_width
                    source = row * tile_width * components
                    if planar == 2:
                        result[band][destination:destination + cols] = values[source:source + cols]
                    else:
                        for col in range(cols):
                            base = source + col * samples
                            for sample in range(samples):
                                result[sample][destination + col] = values[base + sample]
    return result


def _geo_tags(tags: dict[int, Any]) -> dict[str, Any]:
    geo: dict[str, Any] = {}
    for tag, name in (
        (_MODEL_PIXEL_SCALE, "ModelPixelScaleTag"),
        (_MODEL_TIEPOINT, "ModelTiepointTag"),
        (_MODEL_TRANSFORMATION, "ModelTransformationTag"),
    ):
        value = tags.get(tag)
        if isinstance(value, (list, tuple)) and value:
            geo[name] = [float(item) for item in value]
    keys = tags.get(_GEO_KEY_DIRECTORY)
    if isinstance(keys, (list, tuple)) and keys:
        geo["GeoKeyDirectoryTag"] = [int(value) for value in keys]
    return geo


def _nodata_value(tags: dict[int, Any]) -> float | None:
    raw = tags.get(_GDAL_NODATA)
    if isinstance(raw, bytes):
        text = raw.decode("ascii", "replace")
    elif isinstance(raw, str):
        text = raw
    else:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def read_geotiff(path: str | Path) -> GeoRaster:
    """Read the first image of a GeoTIFF into flat band-major lists.

    Raises :class:`GeoTIFFError` when the file uses a feature this reader does
    not implement (see the module docstring for the supported subset).
    """
    target = Path(path)
    data = target.read_bytes()
    endian, big, ifd_offset = _parse_header(data, target)
    tags, _next_ifd = _read_ifd(data, target, endian, big, ifd_offset)

    width = int(_tag_value(tags, _IMAGE_WIDTH, 0))
    height = int(_tag_value(tags, _IMAGE_LENGTH, 0))
    if width <= 0 or height <= 0:
        raise GeoTIFFError(f"{target}: missing or invalid image dimensions")
    samples = max(1, int(_tag_value(tags, _SAMPLES_PER_PIXEL, 1)))
    bits = _uniform(_tag_list(tags, _BITS_PER_SAMPLE, [8]), samples,
                    "BitsPerSample", target)
    sample_format = _uniform(_tag_list(tags, _SAMPLE_FORMAT, [1]), samples,
                             "SampleFormat", target)
    compression = int(_tag_value(tags, _COMPRESSION, 1))
    if compression not in (1, 8, 32946):
        raise GeoTIFFError(
            f"{target}: unsupported TIFF compression {compression}; "
            "only uncompressed and Deflate are implemented"
        )
    predictor = int(_tag_value(tags, _PREDICTOR, 1))
    if predictor not in (1, 2):
        raise GeoTIFFError(
            f"{target}: unsupported TIFF predictor {predictor}; only none and "
            "horizontal differencing are implemented"
        )
    if predictor == 2 and sample_format == 3:
        raise GeoTIFFError(
            f"{target}: horizontal differencing is undefined for floating-point samples"
        )
    planar = int(_tag_value(tags, _PLANAR_CONFIGURATION, 1))
    if planar not in (1, 2):
        raise GeoTIFFError(f"{target}: unsupported planar configuration {planar}")
    if samples == 1:
        planar = 1

    charsets = {1: _UNSIGNED_CHAR, 2: _SIGNED_CHAR, 3: _FLOAT_CHAR}
    charset = charsets.get(sample_format)
    if charset is None:
        raise GeoTIFFError(f"{target}: unsupported sample format {sample_format}")
    dtype = charset.get(bits)
    if dtype is None:
        raise GeoTIFFError(
            f"{target}: unsupported {bits}-bit sample format {sample_format}"
        )

    if _TILE_WIDTH in tags or _TILE_OFFSETS in tags:
        bands = _read_tiles(
            data, target, tags, width, height, samples, dtype, bits, predictor,
            sample_format == 2, planar, compression, endian,
        )
    else:
        bands = _read_strips(
            data, target, tags, width, height, samples, dtype, bits, predictor,
            sample_format == 2, planar, compression, endian,
        )

    return GeoRaster(
        width=width,
        height=height,
        bands=bands,
        band_names=[f"band_{index + 1}" for index in range(samples)],
        nodata=_nodata_value(tags),
        geo_tags=_geo_tags(tags),
    )


# ---------------------------------------------------------------------------
# Scene-level helper
# ---------------------------------------------------------------------------


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(item) for item in paths]


def _check_shapes(rasters: list[GeoRaster]) -> None:
    first = rasters[0]
    for raster in rasters[1:]:
        if (raster.width, raster.height) != (first.width, first.height):
            raise ValueError(
                f"rasters have different shapes: "
                f"{first.width}x{first.height} vs {raster.width}x{raster.height}"
            )


def _collect_bands(rasters: list[GeoRaster],
                   band_names: Sequence[str] | None) -> dict[str, list[float]]:
    names = list(band_names or [])
    bands: dict[str, list[float]] = {}
    total = sum(raster.n_bands for raster in rasters)
    if len(names) > total:
        raise ValueError(f"{len(names)} band names for {total} raster bands")
    if (len(rasters) > 1 and all(raster.n_bands == 1 for raster in rasters)
            and len(names) == len(rasters)):
        for name, raster in zip(names, rasters):
            bands[name] = raster.bands[0]
        return bands
    position = 0
    for raster in rasters:
        for band_index, values in enumerate(raster.bands):
            if position < len(names):
                name = names[position]
            else:
                name = raster.band_names[band_index]
            if name in bands:
                raise ValueError(f"duplicate band name: {name}")
            bands[name] = values
            position += 1
    return bands


def _resample(values: list[float], width: int, height: int, size: int) -> list[float]:
    if width == size and height == size:
        return list(values)
    output: list[float] = []
    for row in range(size):
        source_row = min(height - 1, row * height // size)
        offset = source_row * width
        for col in range(size):
            source_col = min(width - 1, col * width // size)
            output.append(values[offset + source_col])
    return output


def read_geotiff_scene(
    paths: str | Path | Sequence[str | Path],
    band_names: Sequence[str] | None = None,
    labels_path: str | Path | None = None,
    size: int | None = None,
    post_paths: str | Path | Sequence[str | Path] | None = None,
) -> RasterScene:
    """Load one date, or a pre/post pair, plus an optional label raster.

    ``paths`` may be a single file or several files that each contribute one
    band.  Passing ``post_paths`` adds a second date.  When ``size`` is given,
    every band and the labels are nearest-neighbour resampled to
    ``size x size``.  The returned arrays are flat and row-major; building a
    :class:`geocompass.types.Scene` is left to the caller.
    """
    if size is not None and size <= 0:
        raise ValueError("size must be positive")
    pre_files = _as_paths(paths)
    if not pre_files:
        raise ValueError("at least one raster path is required")
    pre_rasters = [read_geotiff(item) for item in pre_files]
    _check_shapes(pre_rasters)
    pre = _collect_bands(pre_rasters, band_names)

    post: dict[str, list[float]] | None = None
    if post_paths is not None:
        post_files = _as_paths(post_paths)
        if not post_files:
            raise ValueError("post_paths is empty")
        post_rasters = [read_geotiff(item) for item in post_files]
        _check_shapes(pre_rasters + post_rasters)
        post = _collect_bands(post_rasters, band_names)

    labels: list[int] | None = None
    if labels_path is not None:
        label_raster = read_geotiff(labels_path)
        if label_raster.n_bands != 1:
            raise ValueError("label raster must have a single band")
        if (label_raster.width, label_raster.height) != (
                pre_rasters[0].width, pre_rasters[0].height):
            raise ValueError("label raster shape does not match the image")
        labels = [int(round(value)) for value in label_raster.bands[0]]

    width = pre_rasters[0].width
    height = pre_rasters[0].height
    if size is not None:
        pre = {name: _resample(values, width, height, size)
               for name, values in pre.items()}
        if post is not None:
            post = {name: _resample(values, width, height, size)
                    for name, values in post.items()}
        if labels is not None:
            labels = [int(value) for value in _resample(
                [float(value) for value in labels], width, height, size)]
        width = height = size

    return RasterScene(
        pre=pre,
        post=post,
        labels=labels,
        width=width,
        height=height,
        nodata=pre_rasters[0].nodata,
        geo_tags=dict(pre_rasters[0].geo_tags),
    )


# ---------------------------------------------------------------------------
# Test-only writer
# ---------------------------------------------------------------------------

_WRITE_FORMATS = {
    "uint16": (16, 1, "H"),
    "float32": (32, 3, "f"),
}


def _encode_entries(endian: str, entries: list[tuple[int, int, list[Any]]],
                    ifd_offset: int) -> tuple[bytes, bytes]:
    entry_size = 12
    ifd_size = 2 + entry_size * len(entries) + 4
    extra_base = ifd_offset + ifd_size
    extra = bytearray()
    body = bytearray()
    for tag, ttype, values in entries:
        char = _TYPE_CHAR[ttype]
        raw = struct.pack(endian + str(len(values)) + char, *values)
        if len(raw) <= 4:
            value_field = raw.ljust(4, b"\x00")
        else:
            value_field = struct.pack(endian + "I", extra_base + len(extra))
            extra.extend(raw)
            if len(extra) % 2:
                extra.append(0)
        body.extend(struct.pack(endian + "HHI", tag, ttype, len(values)))
        body.extend(value_field)
    ifd = struct.pack(endian + "H", len(entries)) + bytes(body)
    ifd += struct.pack(endian + "I", 0)
    return ifd, bytes(extra)


def write_geotiff(
    path: str | Path,
    rasters: Sequence[Sequence[float]] | GeoRaster,
    *,
    width: int | None = None,
    height: int | None = None,
    sample_format: str = "uint16",
    rows_per_strip: int | None = None,
    byteorder: str = "<",
    geo_tags: dict[str, Any] | None = None,
) -> None:
    """Write a minimal uncompressed classic GeoTIFF.

    Test utility only: it exists so the reader can be exercised without
    external data.  It writes chunky samples, a single IFD and no compression,
    which is far below the feature level of a real GeoTIFF writer.  Supported
    sample formats are ``"uint16"`` and ``"float32"``; ``byteorder`` may be
    ``"<"`` (``II``) or ``">"`` (``MM``).  ``rasters`` is either band-major
    data plus explicit ``width``/``height``, or a :class:`GeoRaster`.
    """
    if sample_format not in _WRITE_FORMATS:
        raise ValueError(f"unsupported sample format: {sample_format}")
    if byteorder not in ("<", ">"):
        raise ValueError("byteorder must be '<' or '>'")

    if isinstance(rasters, GeoRaster):
        bands = [list(band) for band in rasters.bands]
        width = rasters.width if width is None else width
        height = rasters.height if height is None else height
    else:
        bands = [list(band) for band in rasters]
    if not bands:
        raise ValueError("at least one band is required")
    if width is None or height is None:
        raise ValueError("width and height are required")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    expected = width * height
    for band in bands:
        if len(band) != expected:
            raise ValueError(f"band has {len(band)} samples, expected {expected}")

    bits, sample_format_tag, code = _WRITE_FORMATS[sample_format]
    if sample_format == "uint16":
        encoded: list[list[int | float]] = []
        for band in bands:
            values = []
            for value in band:
                item = int(round(value))
                if item < 0 or item > 65535:
                    raise ValueError("uint16 sample out of range")
                values.append(item)
            encoded.append(values)
    else:
        encoded = [[float(value) for value in band] for band in bands]

    samples = len(bands)
    rows_per_strip = height if rows_per_strip is None else rows_per_strip
    if rows_per_strip <= 0:
        raise ValueError("rows_per_strip must be positive")
    rows_per_strip = min(rows_per_strip, height)
    strips = (height + rows_per_strip - 1) // rows_per_strip

    payload = bytearray()
    offsets: list[int] = []
    counts: list[int] = []
    for strip in range(strips):
        row0 = strip * rows_per_strip
        rows = min(rows_per_strip, height - row0)
        offsets.append(8 + len(payload))
        flat: list[int | float] = []
        for row in range(rows):
            base = (row0 + row) * width
            for col in range(width):
                pixel = base + col
                for band in range(samples):
                    flat.append(encoded[band][pixel])
        chunk = struct.pack(byteorder + str(len(flat)) + code, *flat)
        payload.extend(chunk)
        counts.append(len(chunk))

    entries: list[tuple[int, int, list[Any]]] = [
        (_IMAGE_WIDTH, 4, [width]),
        (_IMAGE_LENGTH, 4, [height]),
        (_BITS_PER_SAMPLE, 3, [bits] * samples),
        (_COMPRESSION, 3, [1]),
        (_PHOTOMETRIC, 3, [2 if samples >= 3 else 1]),
        (_STRIP_OFFSETS, 4, offsets),
        (_SAMPLES_PER_PIXEL, 3, [samples]),
        (_ROWS_PER_STRIP, 4, [rows_per_strip]),
        (_STRIP_BYTE_COUNTS, 4, counts),
        (_PLANAR_CONFIGURATION, 3, [1]),
        (_SAMPLE_FORMAT, 3, [sample_format_tag] * samples),
    ]
    for key, value in (geo_tags or {}).items():
        tag = _GEO_TAG_ALIASES.get(key) if isinstance(key, str) else key
        if tag is None:
            raise ValueError(f"unknown geo tag: {key}")
        values = [float(item) for item in value]
        if tag == _MODEL_PIXEL_SCALE and len(values) != 3:
            raise ValueError("ModelPixelScaleTag needs three values")
        if tag == _MODEL_TIEPOINT and len(values) % 6:
            raise ValueError("ModelTiepointTag needs a multiple of six values")
        if tag == _MODEL_TRANSFORMATION and len(values) != 16:
            raise ValueError("ModelTransformationTag needs sixteen values")
        if tag == _GEO_KEY_DIRECTORY:
            entries.append((tag, 3, [int(item) for item in values]))
        else:
            entries.append((tag, 12, values))

    entries.sort(key=lambda item: item[0])
    ifd_offset = 8 + len(payload)
    ifd, extra = _encode_entries(byteorder, entries, ifd_offset)
    header = (b"II" if byteorder == "<" else b"MM")
    header += struct.pack(byteorder + "HI", 42, ifd_offset)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + bytes(payload) + ifd + extra)
