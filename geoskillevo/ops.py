"""Primitive raster operations and grading used by skills."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .types import BANDS, OpSpec, Scene, Skill, TaskSpec


class Incompatible(ValueError):
    """Raised when a skill cannot run on a task."""


@dataclass
class Context:
    task: TaskSpec
    scene: Scene
    continuous: list[float] | None = None
    map: list[int] | None = None
    trace: list[str] = field(default_factory=list)
    ops: int = 0

    @property
    def size(self) -> int:
        return self.scene.size

    def require_field(self) -> list[float]:
        if self.continuous is None:
            raise Incompatible("operation needs a continuous field")
        return self.continuous

    def require_map(self) -> list[int]:
        if self.map is None:
            raise Incompatible("operation needs a thematic map")
        return self.map


def _field_from_bands(scene: Scene, date: str, band: str) -> list[float]:
    data = scene.post if date == "post" else scene.pre
    if band not in data:
        raise Incompatible(f"missing band: {band}")
    return data[band]


def compute_index(scene: Scene, name: str, date: str = "post") -> list[float]:
    """Compute a normalized-difference index for one date."""
    size = scene.n_pixels

    def band(key: str) -> list[float]:
        return _field_from_bands(scene, date, key)

    if name in {"ndvi", "ndwi", "ndbi", "nbr", "ndmi"}:
        pairs = {
            "ndvi": ("nir", "red"),
            "ndwi": ("green", "nir"),
            "ndbi": ("swir1", "nir"),
            "nbr": ("nir", "swir2"),
            "ndmi": ("nir", "swir1"),
        }
        a, b = pairs[name]
        left, right = band(a), band(b)
        return [(u - v) / max(1e-6, u + v) for u, v in zip(left, right)]
    if name == "bsi":
        swir1, red, nir, blue = band("swir1"), band("red"), band("nir"), band("blue")
        out = []
        for s, r, n, b in zip(swir1, red, nir, blue):
            numerator = (s + r) - (n + b)
            denominator = (s + r) + (n + b)
            out.append(numerator / max(1e-6, denominator))
        return out
    if name == "brightness":
        fields = [band(key) for key in BANDS]
        return [sum(values) / len(values) for values in zip(*fields)]
    if name == "evi":
        nir, red, blue = band("nir"), band("red"), band("blue")
        return [
            2.5 * (n - r) / max(1e-6, n + 6.0 * r - 7.5 * b + 1.0)
            for n, r, b in zip(nir, red, blue)
        ]
    if name.startswith("band:"):
        return list(band(name.split(":", 1)[1]))
    raise Incompatible(f"unknown index: {name}")


def otsu_threshold(values: list[float], bins: int = 64) -> float:
    """Otsu threshold on a continuous field."""
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        raise Incompatible("empty field")
    low, high = min(clean), max(clean)
    if high - low < 1e-9:
        return high
    width = (high - low) / bins
    histogram = [0] * bins
    for value in clean:
        index = min(bins - 1, int((value - low) / width))
        histogram[index] += 1
    total = len(clean)
    sum_all = sum((i + 0.5) * width + low for i, count in enumerate(histogram) for _ in range(count))
    weight_bg = 0
    sum_bg = 0.0
    best_score = -1.0
    best = low + width / 2.0
    for i, count in enumerate(histogram):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        center = (i + 0.5) * width + low
        sum_bg += count * center
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        score = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if score > best_score:
            best_score = score
            best = center
    return best


def _majority(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda key: (counts[key], key))


def _neighbours(index: int, size: int, radius: int):
    x, y = index % size, index // size
    for yy in range(max(0, y - radius), min(size, y + radius + 1)):
        for xx in range(max(0, x - radius), min(size, x + radius + 1)):
            yield yy * size + xx


def majority_filter(data: list[int], size: int, radius: int = 1) -> list[int]:
    out = list(data)
    for index in range(len(data)):
        out[index] = _majority([data[j] for j in _neighbours(index, size, radius)])
    return out


def median_filter(data: list[int], size: int, radius: int = 1) -> list[int]:
    out = list(data)
    for index in range(len(data)):
        values = sorted(data[j] for j in _neighbours(index, size, radius))
        out[index] = values[len(values) // 2]
    return out


def _components(data: list[int], size: int, value: int) -> list[list[int]]:
    seen = [False] * len(data)
    components: list[list[int]] = []
    for start, current in enumerate(data):
        if current != value or seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbour in _neighbours(index, size, 1):
                if not seen[neighbour] and data[neighbour] == value:
                    seen[neighbour] = True
                    stack.append(neighbour)
        components.append(component)
    return components


def remove_small(data: list[int], size: int, min_size: int = 12) -> list[int]:
    out = list(data)
    for value in sorted(set(data)):
        if value == 0:
            continue
        for component in _components(out, size, value):
            if len(component) < min_size:
                for index in component:
                    out[index] = 0
    return out


def fill_holes(data: list[int], size: int, max_size: int = 24) -> list[int]:
    out = list(data)
    for component in _components(out, size, 0):
        border = any(index % size in (0, size - 1) or index // size in (0, size - 1)
                     for index in component)
        if not border and len(component) <= max_size:
            for index in component:
                out[index] = 1
    return out


def morph(data: list[int], size: int, op: str = "close", radius: int = 1) -> list[int]:
    binary = [int(value != 0) for value in data]
    if op == "dilate":
        out = [0] * len(binary)
        for index, value in enumerate(binary):
            if value:
                for neighbour in _neighbours(index, size, radius):
                    out[neighbour] = 1
        return out
    if op == "erode":
        out = [1] * len(binary)
        for index in range(len(binary)):
            for neighbour in _neighbours(index, size, radius):
                if not binary[neighbour]:
                    out[index] = 0
                    break
        return out
    if op == "open":
        return morph(morph(binary, size, "erode", radius), size, "dilate", radius)
    if op == "close":
        return morph(morph(binary, size, "dilate", radius), size, "erode", radius)
    raise Incompatible(f"unknown morphology: {op}")


def classify_nearest_centroid(ctx: Context, features: list[str], date: str = "post") -> list[int]:
    scene = ctx.scene
    if not scene.samples:
        raise Incompatible("nearest-centroid classification needs labelled samples")

    def vector(index: int) -> list[float]:
        values = []
        for feature in features:
            if feature.startswith("band:"):
                values.append(_field_from_bands(scene, date, feature.split(":", 1)[1])[index])
            else:
                values.append(compute_index(scene, feature, date)[index])
        return values

    sample_vectors = [vector(index) for index in scene.samples]
    width = len(features)
    means = [sum(vector[j] for vector in sample_vectors) / len(sample_vectors) for j in range(width)]
    stds = []
    for j in range(width):
        variance = sum((vector[j] - means[j]) ** 2 for vector in sample_vectors) / max(1, len(sample_vectors))
        stds.append(max(1e-6, math.sqrt(variance)))
    labels = sorted(set(scene.sample_labels))
    centroids = {}
    for label in labels:
        rows = [vector for vector, truth in zip(sample_vectors, scene.sample_labels) if truth == label]
        if not rows:
            continue
        centroids[label] = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
    if not centroids:
        raise Incompatible("no training classes")
    out = []
    for index in range(scene.n_pixels):
        point = vector(index)
        best_label, best_distance = labels[0], float("inf")
        for label, centroid in centroids.items():
            distance = sum(((point[j] - centroid[j]) / stds[j]) ** 2 for j in range(width))
            if distance < best_distance:
                best_label, best_distance = label, distance
        out.append(best_label)
    return out


def apply_op(ctx: Context, op: OpSpec) -> None:
    name = op.op
    params = op.as_dict()
    params.pop("op", None)
    ctx.ops += 1
    ctx.trace.append(name)
    if name == "index":
        ctx.continuous = compute_index(ctx.scene, params["name"], params.get("date", "post"))
    elif name == "temporal_diff":
        index = params.get("index", "ndvi")
        before = compute_index(ctx.scene, index, "pre")
        after = compute_index(ctx.scene, index, "post")
        if params.get("absolute", True):
            ctx.continuous = [abs(a - b) for a, b in zip(after, before)]
        else:
            ctx.continuous = [a - b for a, b in zip(after, before)]
    elif name == "temporal_ratio":
        index = params.get("index", "ndvi")
        before = compute_index(ctx.scene, index, "pre")
        after = compute_index(ctx.scene, index, "post")
        ctx.continuous = [abs((a + 0.2) / (b + 0.2) - 1.0) for a, b in zip(after, before)]
    elif name == "spectral_change":
        bands = list(params.get("bands", BANDS))
        before = [ctx.scene.pre[band] for band in bands]
        after = [ctx.scene.post[band] for band in bands]
        ctx.continuous = [
            math.sqrt(sum((after[k][i] - before[k][i]) ** 2 for k in range(len(bands))))
            for i in range(ctx.scene.n_pixels)
        ]
    elif name == "threshold":
        values = ctx.require_field()
        method = params.get("method", "otsu")
        if method == "otsu":
            threshold = otsu_threshold(values, int(params.get("bins", 64)))
        elif method == "percentile":
            quantile = float(params.get("quantile", 90.0))
            ordered = sorted(values)
            position = min(len(ordered) - 1, max(0, int(round(quantile / 100.0 * (len(ordered) - 1)))))
            threshold = ordered[position]
        elif method == "mean_sd":
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            threshold = mean + float(params.get("k", 1.0)) * math.sqrt(variance)
        else:
            threshold = float(params.get("value", 0.0))
        direction = params.get("direction", "high")
        if direction == "high":
            ctx.map = [int(value > threshold) for value in values]
        else:
            ctx.map = [int(value < threshold) for value in values]
    elif name == "classify":
        ctx.map = classify_nearest_centroid(ctx, list(params["features"]), params.get("date", "post"))
    elif name == "smooth":
        data = ctx.require_map()
        method = params.get("method", "majority")
        radius = int(params.get("radius", 1))
        if method == "median":
            ctx.map = median_filter(data, ctx.size, radius)
        else:
            ctx.map = majority_filter(data, ctx.size, radius)
    elif name == "remove_small":
        ctx.map = remove_small(ctx.require_map(), ctx.size, int(params.get("min_size", 12)))
    elif name == "fill_holes":
        ctx.map = fill_holes(ctx.require_map(), ctx.size, int(params.get("max_size", 24)))
    elif name == "morph":
        ctx.map = morph(ctx.require_map(), ctx.size, params.get("mode", "close"), int(params.get("radius", 1)))
    else:
        raise Incompatible(f"unknown operation: {name}")


def execute_skill(skill: Skill, task: TaskSpec, scene: Scene) -> Context:
    if skill.input_kind == "bitemporal" and not task.bitemporal:
        raise Incompatible("skill needs a bi-temporal task")
    if skill.input_kind == "single" and task.bitemporal and skill.kind == "analysis":
        # Single-date analysis is still runnable on the post image, but the flag
        # is recorded so the planner can prefer temporal evidence.
        pass
    missing = [band for band in skill.required_bands if band not in scene.post]
    if missing:
        raise Incompatible(f"missing bands: {missing}")
    ctx = Context(task=task, scene=scene)
    for op in skill.ops:
        apply_op(ctx, op)
    if ctx.map is None:
        raise Incompatible("skill produced no map")
    if len(ctx.map) != scene.n_pixels:
        raise Incompatible("skill produced a map with the wrong shape")
    return ctx


def grade(scene: Scene, prediction: list[int]) -> dict[str, float]:
    truth = scene.truth
    if len(prediction) != len(truth):
        raise ValueError("prediction and truth differ in length")
    if scene.truth_kind == "multiclass":
        correct = sum(int(a == b) for a, b in zip(prediction, truth))
        accuracy = correct / len(truth)
        labels = sorted(set(truth))
        f1s = []
        for label in labels:
            tp = sum(1 for a, b in zip(prediction, truth) if a == label and b == label)
            fp = sum(1 for a, b in zip(prediction, truth) if a == label and b != label)
            fn = sum(1 for a, b in zip(prediction, truth) if a != label and b == label)
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        return {
            "score": accuracy,
            "oa": accuracy,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "iou": sum(
                sum(1 for a, b in zip(prediction, truth) if a == label and b == label)
                / max(1, sum(1 for a, b in zip(prediction, truth) if a == label or b == label))
                for label in labels
            ) / len(labels),
        }
    predicted = [int(value != 0) for value in prediction]
    tp = sum(1 for a, b in zip(predicted, truth) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(predicted, truth) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(predicted, truth) if a == 0 and b == 1)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    accuracy = sum(1 for a, b in zip(predicted, truth) if a == b) / len(truth)
    return {"score": f1, "oa": accuracy, "f1": f1, "iou": iou}


def answer_accuracy(scene: Scene, prediction: list[int]) -> float:
    return grade(scene, prediction)["score"]
