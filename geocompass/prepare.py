"""Turn a public chip collection into the file layout the manifest expects.

The loaders do not know about any particular collection.  What a collection
needs is a small conversion step: select the bands the skill library uses,
rescale reflectance, remap the label encoding, and write a manifest.

Only the pieces that can be verified without the archive in hand live here;
the script that walks a downloaded directory sits in ``scripts/`` and calls
them.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .io import GeoRaster, read_geotiff, write_geotiff
from .types import BANDS

# Sen1Floods11 stores 13 Sentinel-2 bands in acquisition order.
SEN1FLOODS11_BANDS = (1, 2, 3, 7, 11, 12)


class PrepareError(ValueError):
    """Raised when a source chip cannot be mapped onto the band contract."""


@dataclass
class Chip:
    """One converted chip pair."""

    identifier: str
    image: str
    labels: str
    family: str
    split: str


def _select(raster: GeoRaster, indices: Sequence[int]) -> list[list[float]]:
    if len(indices) != len(BANDS):
        raise PrepareError(f"expected {len(BANDS)} band indices, got {len(indices)}")
    missing = [index for index in indices if index >= len(raster.bands)]
    if missing:
        raise PrepareError(
            f"{raster.bands and len(raster.bands)} bands in source, cannot select {missing}")
    return [raster.bands[index] for index in indices]


def convert_image(source: str | Path, destination: str | Path,
                  indices: Sequence[int] = SEN1FLOODS11_BANDS,
                  scale: float = 1e-4, offset: float = 0.0) -> Path:
    """Write a six-band float32 chip from a multi-band source."""
    raster = read_geotiff(source)
    bands = _select(raster, indices)
    if scale != 1.0 or offset != 0.0:
        bands = [[value * scale + offset for value in band] for band in bands]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(destination, bands, width=raster.width, height=raster.height,
                  sample_format="float32")
    return destination


def convert_labels(source: str | Path, destination: str | Path,
                   nodata: int = -1, replacement: int = 65535) -> Path:
    """Remap an encoded label raster to the benchmark's 0/1 (or class) codes."""
    raster = read_geotiff(source)
    if len(raster.bands) != 1:
        raise PrepareError(f"{source}: label raster must have one band")
    values = [replacement if int(round(value)) == nodata else int(round(value))
              for value in raster.bands[0]]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(destination, [values], width=raster.width, height=raster.height,
                  sample_format="uint16")
    return destination


def manifest_payload(chips: Iterable[Chip], name: str, size: int | None = 32,
                     scale: float = 1.0) -> dict:
    entries = []
    for chip in chips:
        entry = {
            "id": chip.identifier,
            "family": chip.family,
            "split": chip.split,
            "pre": [chip.image],
            "labels": chip.labels,
        }
        if size:
            entry["size"] = size
        entries.append(entry)
    return {
        "name": name,
        "band_names": list(BANDS),
        "scale": scale,
        "size": size,
        "entries": entries,
    }


def write_manifest(payload: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
