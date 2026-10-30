"""Synthetic multi-band spatiotemporal scene generator.

The generator is intentionally small (32 x 32 pixels by default) and deterministic.  It
produces realistic-enough spectral contrast for land-cover mapping and
bi-temporal change detection while keeping every experiment CPU-only and
reproducible.  Real GeoTIFF chips enter through :mod:`geocompass.realdata`.
"""
from __future__ import annotations

import random

from .types import BANDS, CLASS_INDEX, CLASS_NAMES, Scene, TaskSpec

# Mean surface reflectance per class and band (blue, green, red, nir, swir1, swir2).
SPECTRA = {
    "water": (0.055, 0.075, 0.045, 0.020, 0.025, 0.018),
    "forest": (0.030, 0.055, 0.040, 0.360, 0.145, 0.065),
    "grass": (0.045, 0.090, 0.060, 0.280, 0.230, 0.150),
    "crop": (0.040, 0.085, 0.045, 0.460, 0.190, 0.100),
    "urban": (0.125, 0.145, 0.155, 0.185, 0.255, 0.285),
    "bare": (0.155, 0.185, 0.225, 0.255, 0.330, 0.390),
}

SEASON_GAIN = {
    "spring": (1.10, 1.05, 0.95, 0.92, 0.98, 1.00),
    "summer": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "fall": (1.05, 0.98, 1.10, 0.88, 1.06, 1.08),
    "winter": (1.12, 1.05, 1.15, 0.80, 1.10, 1.12),
}

REGION_BIAS = {
    "central": {"water": 1.0, "forest": 1.0, "grass": 1.0, "crop": 1.0, "urban": 1.0, "bare": 1.0},
    "north": {"water": 0.8, "forest": 1.7, "grass": 1.2, "crop": 0.6, "urban": 0.8, "bare": 0.5},
    "south": {"water": 0.7, "forest": 0.5, "grass": 0.9, "crop": 1.6, "urban": 1.0, "bare": 1.5},
    "coast": {"water": 2.0, "forest": 0.9, "grass": 0.9, "crop": 0.8, "urban": 1.2, "bare": 0.8},
}


def _clip(value: float, low: float = 0.001, high: float = 0.95) -> float:
    return max(low, min(high, value))


def _ellipse(grid: list[int], size: int, cls: int, cx: int, cy: int, rx: int, ry: int,
             rng: random.Random, fill: float = 0.94) -> None:
    for y in range(max(0, cy - ry), min(size, cy + ry + 1)):
        for x in range(max(0, cx - rx), min(size, cx + rx + 1)):
            dx = (x - cx) / max(1, rx)
            dy = (y - cy) / max(1, ry)
            if dx * dx + dy * dy <= 1.0 and rng.random() < fill:
                grid[y * size + x] = cls


def _rect(grid: list[int], size: int, cls: int, x0: int, y0: int, w: int, h: int,
          rng: random.Random, fill: float = 0.97) -> None:
    for y in range(max(0, y0), min(size, y0 + h)):
        for x in range(max(0, x0), min(size, x0 + w)):
            if rng.random() < fill:
                grid[y * size + x] = cls


def _class_map(spec: TaskSpec, rng: random.Random) -> list[int]:
    size = spec.size
    grid = [CLASS_INDEX["grass"]] * (size * size)
    bias = REGION_BIAS.get(spec.region, REGION_BIAS["central"])

    def count(name: str) -> float:
        return sum(1 for value in grid if value == CLASS_INDEX[name]) / len(grid)

    def place(name: str, target: float, attempts: int, rx: tuple[int, int],
              ry: tuple[int, int], center: bool = False) -> None:
        tries = 0
        while count(name) < target and tries < attempts:
            tries += 1
            if center:
                cx = rng.randint(size // 4, 3 * size // 4)
                cy = rng.randint(size // 4, 3 * size // 4)
            else:
                cx, cy = rng.randrange(size), rng.randrange(size)
            _ellipse(grid, size, CLASS_INDEX[name], cx, cy,
                     rng.randint(*rx), rng.randint(*ry), rng)

    place("forest", 0.24 * bias["forest"], 8, (size // 10, size // 4), (size // 12, size // 4))
    # Agricultural fields: a few rectangles.
    fields = max(1, int(round(4 * min(1.6, bias["crop"]))))
    for _ in range(fields):
        w = rng.randint(size // 8, size // 3)
        h = rng.randint(size // 8, size // 3)
        x0 = rng.randrange(max(1, size - w))
        y0 = rng.randrange(max(1, size - h))
        _rect(grid, size, CLASS_INDEX["crop"], x0, y0, w, h, rng)
    place("urban", 0.11 * bias["urban"], 5, (size // 14, size // 8), (size // 14, size // 8), center=True)
    place("bare", 0.08 * bias["bare"], 5, (size // 14, size // 7), (size // 14, size // 7))
    # Guarantee the non-water classes are represented.
    for name in ("forest", "crop", "urban", "bare"):
        cls = CLASS_INDEX[name]
        if sum(1 for value in grid if value == cls) < max(12, size):
            _ellipse(grid, size, cls, rng.randrange(size), rng.randrange(size),
                     max(2, size // 16), max(2, size // 16), rng)
    # Water is placed last so that narrow rivers and lakes survive later writes.
    place("water", 0.09 * bias["water"], 8, (size // 10, size // 3), (size // 12, size // 4))
    if count("water") < 0.02:
        _ellipse(grid, size, CLASS_INDEX["water"], size // 2, size // 2,
                 max(3, size // 8), max(3, size // 10), rng)
    return grid


def _apply_change(spec: TaskSpec, classes: list[int], rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    """Return post classes, generic change mask, and burn mask."""
    size = spec.size
    post = list(classes)
    burn = [0] * (size * size)
    change_type = spec.change_type
    if change_type == "mixed":
        change_type = rng.choice(["urban", "deforest", "rotation", "flood"])

    def set_class(index: int, name: str) -> None:
        nonlocal post
        post[index] = CLASS_INDEX[name]

    if change_type == "urban":
        urban = CLASS_INDEX["urban"]
        candidates = [
            i for i, value in enumerate(classes)
            if value != urban and any(
                classes[j] == urban
                for j in (
                    i - 1 if i % size else i,
                    i + 1 if (i + 1) % size else i,
                    i - size if i >= size else i,
                    i + size if i + size < len(classes) else i,
                )
            )
        ]
        rng.shuffle(candidates)
        for index in candidates[: max(20, int(0.035 * size * size))]:
            set_class(index, "urban")
    elif change_type == "deforest":
        cx, cy = rng.randrange(size), rng.randrange(size)
        rx, ry = rng.randint(size // 10, size // 5), rng.randint(size // 10, size // 5)
        for y in range(max(0, cy - ry), min(size, cy + ry + 1)):
            for x in range(max(0, cx - rx), min(size, cx + rx + 1)):
                dx, dy = (x - cx) / max(1, rx), (y - cy) / max(1, ry)
                if dx * dx + dy * dy <= 1.0 and rng.random() < 0.9:
                    index = y * size + x
                    if classes[index] == CLASS_INDEX["forest"]:
                        set_class(index, rng.choice(["bare", "grass", "bare"]))
    elif change_type == "rotation":
        # Swap blocks of crop and grass to mimic field rotation.
        for _ in range(3):
            w, h = rng.randint(size // 8, size // 3), rng.randint(size // 8, size // 3)
            x0, y0 = rng.randrange(max(1, size - w)), rng.randrange(max(1, size - h))
            for y in range(y0, y0 + h):
                for x in range(x0, x0 + w):
                    index = y * size + x
                    if classes[index] == CLASS_INDEX["crop"]:
                        set_class(index, "grass")
                    elif classes[index] == CLASS_INDEX["grass"]:
                        set_class(index, "crop")
    elif change_type == "flood":
        water = CLASS_INDEX["water"]
        # Expand existing water into low-lying neighbouring classes.
        for _ in range(max(15, int(0.02 * size * size))):
            sources = [i for i, value in enumerate(post) if value == water]
            if not sources:
                break
            index = rng.choice(sources)
            x, y = index % size, index // size
            neighbours = []
            if x > 0:
                neighbours.append(index - 1)
            if x + 1 < size:
                neighbours.append(index + 1)
            if y > 0:
                neighbours.append(index - size)
            if y + 1 < size:
                neighbours.append(index + size)
            target = rng.choice(neighbours)
            if post[target] != water:
                set_class(target, "water")
    elif change_type == "burn":
        cx, cy = rng.randrange(size), rng.randrange(size)
        rx, ry = rng.randint(size // 12, size // 5), rng.randint(size // 12, size // 5)
        for y in range(max(0, cy - ry), min(size, cy + ry + 1)):
            for x in range(max(0, cx - rx), min(size, cx + rx + 1)):
                dx, dy = (x - cx) / max(1, rx), (y - cy) / max(1, ry)
                if dx * dx + dy * dy <= 1.0 and rng.random() < 0.92:
                    index = y * size + x
                    if classes[index] in (CLASS_INDEX["forest"], CLASS_INDEX["grass"], CLASS_INDEX["crop"]):
                        set_class(index, "bare")
                        burn[index] = 1
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown change type: {change_type}")

    change = [int(a != b) for a, b in zip(classes, post)]
    if sum(change) < max(8, int(0.005 * size * size)):
        # Guarantee a non-degenerate target.  A small urban/bare patch is
        # inserted deterministically rather than returning an all-zero mask.
        cx, cy = rng.randrange(size), rng.randrange(size)
        for y in range(max(0, cy - 2), min(size, cy + 3)):
            for x in range(max(0, cx - 2), min(size, cx + 3)):
                index = y * size + x
                if classes[index] != CLASS_INDEX["urban"]:
                    post[index] = CLASS_INDEX["urban"]
                    if post[index] != classes[index]:
                        change[index] = 1
    return post, change, burn


def _smooth_noise(size: int, rng: random.Random, scale: float) -> list[float]:
    coarse = [[rng.gauss(0.0, scale) for _ in range(size // 4 + 2)] for _ in range(size // 4 + 2)]

    def sample(x: float, y: float) -> float:
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        a = coarse[y0][x0] * (1 - fx) + coarse[y0][x0 + 1] * fx
        b = coarse[y0 + 1][x0] * (1 - fx) + coarse[y0 + 1][x0 + 1] * fx
        return a * (1 - fy) + b * fy

    return [sample(x / 4.0, y / 4.0) for y in range(size) for x in range(size)]


def _bands(classes: list[int], spec: TaskSpec, rng: random.Random, date: str,
           burn: list[int] | None = None) -> dict[str, list[float]]:
    size = spec.size
    gains = [rng.gauss(1.0, 0.035) for _ in BANDS]
    season = SEASON_GAIN.get(date if date in SEASON_GAIN else spec.season, SEASON_GAIN["summer"])
    textures = {band: _smooth_noise(size, rng, 0.012) for band in BANDS}
    fields: dict[str, list[float]] = {band: [0.0] * (size * size) for band in BANDS}
    for index, cls in enumerate(classes):
        name = CLASS_NAMES[cls]
        base = SPECTRA[name]
        local = rng.gauss(1.0, 0.06)
        for band_index, band in enumerate(BANDS):
            value = base[band_index] * gains[band_index] * season[band_index] * local
            if burn is not None and burn[index]:
                value *= (0.62 if band in {"nir", "swir1", "swir2"} else 0.72)
            value += textures[band][index] + rng.gauss(0.0, 0.010)
            fields[band][index] = _clip(value)
    # Cloud contamination: bright, spatially contiguous masks.
    if spec.cloud > 0:
        cloud = [0] * (size * size)
        for _ in range(max(1, int(spec.cloud * 8))):
            cx, cy = rng.randrange(size), rng.randrange(size)
            rx, ry = rng.randint(2, max(3, size // 6)), rng.randint(2, max(3, size // 6))
            _ellipse(cloud, size, 1, cx, cy, rx, ry, rng, fill=0.8)
        for index, flag in enumerate(cloud):
            if flag:
                for band in ("blue", "green", "red"):
                    fields[band][index] = _clip(fields[band][index] + rng.uniform(0.18, 0.34), high=0.98)
                fields["nir"][index] = _clip(fields["nir"][index] * 0.85)
    return fields


def _truth(spec: TaskSpec, scene: Scene, burn: list[int]) -> tuple[list[int], str]:
    family = spec.family
    if family == "landcover":
        return list(scene.classes_post), "multiclass"
    if family == "crop":
        return [int(value == CLASS_INDEX["crop"]) for value in scene.classes_post], "binary"
    if family in {"water", "flood"}:
        return [int(value == CLASS_INDEX["water"]) for value in scene.classes_post], "binary"
    if family == "change":
        return list(scene.change_mask), "binary"
    if family == "urban":
        return [
            int(post == CLASS_INDEX["urban"] and pre != CLASS_INDEX["urban"])
            for pre, post in zip(scene.classes_pre, scene.classes_post)
        ], "binary"
    if family == "burn":
        return list(burn), "binary"
    raise ValueError(f"unsupported family: {family}")


def _samples(scene: Scene, truth: list[int], rng: random.Random, per_class: int = 14) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    labels: list[int] = []
    for label in sorted(set(truth)):
        pool = [i for i, value in enumerate(truth) if value == label and scene.valid[i]]
        rng.shuffle(pool)
        chosen = pool[:per_class]
        indices.extend(chosen)
        labels.extend([label] * len(chosen))
    order = list(range(len(indices)))
    rng.shuffle(order)
    return [indices[i] for i in order], [labels[i] for i in order]


def generate_scene(spec: TaskSpec, size: int | None = None) -> Scene:
    """Generate a deterministic scene for ``spec``."""
    if size is not None and size != spec.size:
        raise ValueError("scene size must match spec.size")
    size = spec.size
    rng = random.Random(1000003 * (spec.seed + 1) + sum(ord(c) for c in spec.task_id))
    classes_pre = _class_map(spec, rng)
    if spec.bitemporal:
        classes_post, change, burn = _apply_change(spec, classes_pre, rng)
        pre = _bands(classes_pre, spec, rng, "pre")
        post = _bands(classes_post, spec, rng, "post", burn)
    else:
        classes_post = list(classes_pre)
        change = [0] * (size * size)
        burn = [0] * (size * size)
        pre = _bands(classes_pre, spec, rng, "pre")
        post = _bands(classes_post, spec, rng, "post")
    valid = [1] * (size * size)
    scene = Scene(
        size=size,
        pre=pre,
        post=post,
        classes_pre=classes_pre,
        classes_post=classes_post,
        change_mask=change,
        valid=valid,
        truth=[],
        truth_kind="binary",
        samples=[],
        sample_labels=[],
        change_type=spec.change_type,
        cloud_fraction=spec.cloud,
    )
    truth, kind = _truth(spec, scene, burn)
    scene.truth = truth
    scene.truth_kind = kind
    scene.samples, scene.sample_labels = _samples(scene, truth, rng)
    return scene


def load_geotiff(path, band_names=None, labels_path=None, size=None,
                 post_paths=None):
    """Compatibility shim for the real-data adapter.

    The benchmark stays dependency-free: ``geocompass.io`` decodes the file and
    ``geocompass.realdata`` maps it onto a :class:`~geocompass.types.Scene`.
    See ``docs/REAL_DATA.md`` for the contract and the supported encodings.
    """
    from .realdata import load_scene

    return load_scene(path, band_names=band_names, labels_path=labels_path,
                      size=size, post_paths=post_paths)


def ndvi(red: float, nir: float) -> float:
    return (nir - red) / max(1e-6, nir + red)
