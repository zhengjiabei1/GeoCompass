"""Build benchmark scenes from real Earth-observation rasters.

``geocompass.io`` decodes GeoTIFF chips; this module maps them onto the same
``Scene`` structure the synthetic generator produces, so that the selector,
transfer diagnosis, and gating code run unchanged on real data.

The adapter makes the data contract explicit instead of hiding it:

* all six optical bands must be present for the spectral rules to be
  meaningful, and a missing band is an error rather than a silent zero;
* reflectance is scaled and offset by the caller, because the correct factors
  depend on the product level (for example Sentinel-2 L2A stores integer
  reflectance at 1e-4);
* ``nodata`` becomes an invalid-cell mask, and no skill is asked to grade
  pixels that carry no observation;
* labelled reference pixels become the ``samples``/``sample_labels`` pair used
  by the supervised skills.

The module does not download anything.  ``docs/REAL_DATA.md`` describes how to
build a directory of chips from a public collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .io import RasterScene, read_geotiff_scene
from .types import BANDS, Scene


class MissingBandError(ValueError):
    """Raised when a required spectral band is absent from the input."""


@dataclass
class SceneSpec:
    """Everything the adapter needs that cannot be inferred from the pixels."""

    family: str = "water"
    target: str = "water"
    change_type: str = "mixed"
    truth_kind: str = "binary"
    scale: float = 1.0
    offset: float = 0.0
    size: int | None = None


def _scaled(values: list[float], scale: float, offset: float) -> list[float]:
    if scale == 1.0 and offset == 0.0:
        return values
    return [value * scale + offset for value in values]


def _valid_mask(rasters: Sequence[list[float]], nodata: float | None) -> list[int]:
    if nodata is None:
        return [1] * len(rasters[0])
    mask = []
    for index in range(len(rasters[0])):
        cell = 1
        for values in rasters:
            if values[index] == nodata:
                cell = 0
                break
        mask.append(cell)
    return mask


def scene_from_rasters(rasters: RasterScene, spec: SceneSpec | None = None,
                       band_names: Sequence[str] | None = None) -> Scene:
    """Convert decoded rasters into a ``Scene``.

    ``rasters.pre`` is expected to be keyed by the names passed to
    ``read_geotiff_scene``.  Missing bands raise ``MissingBandError`` so that a
    malformed chip cannot silently degrade into a coverage artefact.
    """
    spec = spec or SceneSpec()
    names = list(band_names) if band_names else list(rasters.pre)
    pre = {name: _scaled(rasters.pre[name], spec.scale, spec.offset) for name in names}
    missing = [band for band in BANDS if band not in pre]
    if missing:
        raise MissingBandError(
            f"missing spectral bands {missing}; expected {list(BANDS)}")
    post = None
    if rasters.post is not None:
        post = {name: _scaled(rasters.post[name], spec.scale, spec.offset) for name in names}
    else:
        post = {name: list(values) for name, values in pre.items()}

    width = rasters.width
    height = rasters.height
    size = spec.size or max(width, height)
    valid = _valid_mask([pre[band] for band in BANDS], rasters.nodata)

    labels = rasters.labels
    if labels is None:
        truth = [0] * (width * height)
        truth_kind = spec.truth_kind
    else:
        truth = list(labels)
        truth_kind = "multiclass" if max(truth) > 1 else "binary"

    change_mask = [
        1 if post["nir"][index] != pre["nir"][index] else 0
        for index in range(width * height)
    ] if rasters.post is not None else [0] * (width * height)

    samples = [index for index, cell in enumerate(valid) if cell and labels is not None]
    sample_labels = [int(truth[index]) for index in samples]

    return Scene(
        size=size,
        pre=pre,
        post=post,
        classes_pre=list(truth),
        classes_post=list(truth),
        change_mask=change_mask,
        valid=valid,
        truth=truth,
        truth_kind=truth_kind,
        samples=samples,
        sample_labels=sample_labels,
        change_type=spec.change_type,
        cloud_fraction=0.0,
    )


def load_scene(pre_paths: str | Path | Sequence[str | Path],
               band_names: Sequence[str] | None = None,
               labels_path: str | Path | None = None,
               post_paths: str | Path | Sequence[str | Path] | None = None,
               spec: SceneSpec | None = None,
               size: int | None = None) -> Scene:
    """Read a chip (or a pre/post pair) and build a scene in one call."""
    spec = spec or SceneSpec()
    target_size = size if size is not None else spec.size
    rasters = read_geotiff_scene(pre_paths, band_names=band_names,
                                 labels_path=labels_path, size=target_size,
                                 post_paths=post_paths)
    if target_size is not None:
        spec.size = target_size
    return scene_from_rasters(rasters, spec, band_names=band_names)
