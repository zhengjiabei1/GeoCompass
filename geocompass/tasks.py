"""Task suite, expert skills, and the induced skill library."""
from __future__ import annotations

import hashlib

from .types import (
    BANDS,
    FAMILIES,
    FAMILY_EVIDENCE,
    FAMILY_TARGET,
    FAMILY_TASK_KIND,
    OpSpec,
    Skill,
    TaskSpec,
)

# Acceptance thresholds are family-specific because the metrics differ
# (overall accuracy for land cover, F1 for binary maps).  They were fixed
# before the final test runs.
SUCCESS_THRESHOLD = {
    "landcover": 0.78,
    "crop": 0.55,
    "water": 0.65,
    "flood": 0.60,
    "change": 0.40,
    "urban": 0.45,
    "burn": 0.50,
}

CHANGE_TYPES = {
    "landcover": ("mixed",),
    "crop": ("mixed",),
    "water": ("mixed",),
    "flood": ("flood",),
    "change": ("mixed",),
    "urban": ("urban",),
    "burn": ("burn",),
}


def _spec(task_id: str, family: str, split: str, seed: int, index: int) -> TaskSpec:
    season = ("spring", "summer", "fall")[index % 3]
    region = ("central", "north", "south", "coast")[index % 4]
    cloud = (0.0, 0.08, 0.16)[index % 3]
    change = CHANGE_TYPES[family][index % len(CHANGE_TYPES[family])]
    return TaskSpec(
        task_id=task_id,
        family=family,
        split=split,
        seed=seed * 10007 + index * 131 + len(family),
        region=region,
        season=season,
        cloud=cloud,
        resolution=10 if index % 2 == 0 else 20,
        change_type=change,
        input_kind=FAMILY_TASK_KIND[family],
        targets=(FAMILY_TARGET[family],),
        params={"evidence": FAMILY_EVIDENCE[family]},
    )


def build_task_specs(seed: int, per_split: dict[str, int] | None = None) -> dict[str, list[TaskSpec]]:
    """Build disjoint source/target task splits for one replication seed."""
    counts = per_split or {"source_train": 6, "source_val": 2, "target_dev": 2, "target_test": 4}
    splits: dict[str, list[TaskSpec]] = {name: [] for name in counts}
    for split, count in counts.items():
        index = 0
        for family in FAMILIES:
            for _ in range(count):
                task_id = f"{split}-{seed}-{family}-{index:04d}"
                splits[split].append(_spec(task_id, family, split, seed, index))
                index += 1
    return splits


def _analysis_skill(family: str, variant: int) -> Skill:
    bands = ("blue", "green", "red", "nir", "swir1", "swir2")
    if family == "landcover":
        features = ["band:" + band for band in bands]
        if variant % 2 == 0:
            features += ["ndvi", "ndbi"]
        ops = (OpSpec("classify", (("features", features), ("date", "post"))),)
        name = "multispectral nearest-centroid land-cover classifier"
        desc = ("Six-class land-cover classification with a nearest-centroid "
                "classifier over visible, near-infrared and short-wave bands"
                + (" plus NDVI and NDBI" if variant % 2 == 0 else "") + ".")
        targets = ("landcover",)
    elif family == "crop":
        features = (["band:blue", "band:green", "band:red", "band:nir", "band:swir1",
                     "band:swir2", "bsi"] if variant % 2 == 0
                    else ["ndvi", "ndbi", "bsi", "band:swir1"])
        ops = (OpSpec("classify", (("features", features), ("date", "post"))),)
        name = "index-based crop / non-crop classifier"
        desc = ("Binary crop mapping with a nearest-centroid classifier over "
                "vegetation, built-up and bare-soil indices.")
        targets = ("crop",)
    elif family in {"water", "flood"}:
        index = "ndwi" if variant % 3 != 2 else "ndmi"
        if variant % 4 == 3:
            threshold = (("method", "otsu"), ("direction", "high"))
        else:
            threshold = (("method", "fixed"), ("value", 0.0), ("direction", "high"))
        ops = (
            OpSpec("index", (("name", index), ("date", "post"))),
            OpSpec("threshold", threshold),
        )
        name = f"{index.upper()} water extractor"
        desc = (f"Water and flood mapping from post-event {index.upper()} with a "
                "spectral threshold; single-date water rule.")
        targets = ("water", "flood")
    elif family == "change":
        if variant % 4 == 2:
            ops = (
                OpSpec("temporal_diff", (("index", "ndvi"), ("absolute", True))),
                OpSpec("threshold", (("method", "otsu"), ("direction", "high"))),
            )
            name = "bi-temporal NDVI difference change detector"
            desc = ("Binary change detection from the absolute bi-temporal NDVI "
                    "difference with an Otsu threshold.")
        else:
            threshold = (("method", "otsu"),) if variant % 2 == 0 else (
                ("method", "percentile"), ("quantile", 96.0))
            ops = (
                OpSpec("spectral_change", (("bands", list(BANDS)),)),
                OpSpec("threshold", threshold + (("direction", "high"),)),
            )
            name = "multi-band spectral-distance change detector"
            desc = ("Binary change detection from the multi-band spectral "
                    "distance between two dates, thresholded adaptively.")
        targets = ("change",)
    elif family == "urban":
        if variant % 4 != 2:
            ops = (
                OpSpec("spectral_change", (("bands", list(BANDS)),)),
                OpSpec("threshold", (("method", "otsu"), ("direction", "high"))),
            )
            name = "spectral-distance urban-expansion detector"
            desc = ("Urban expansion from the multi-band spectral distance "
                    "between dates with an Otsu threshold.")
        else:
            ops = (
                OpSpec("temporal_diff", (("index", "ndbi"), ("absolute", False))),
                OpSpec("threshold", (("method", "percentile"), ("quantile", 94.0),
                                     ("direction", "high"))),
            )
            name = "NDBI urban-expansion detector"
            desc = ("Urban expansion from a positive bi-temporal NDBI "
                    "difference with an adaptive threshold.")
        targets = ("urban_change",)
    elif family == "burn":
        if variant % 2 == 0:
            ops = (
                OpSpec("spectral_change", (("bands", list(BANDS)),)),
                OpSpec("threshold", (("method", "otsu"), ("direction", "high"))),
            )
            name = "spectral-distance burn-scar detector"
            desc = ("Burn-scar mapping from the multi-band spectral distance "
                    "between dates with an Otsu threshold.")
        else:
            ops = (
                OpSpec("temporal_diff", (("index", "nbr"), ("absolute", True))),
                OpSpec("threshold", (("method", "percentile"), ("quantile", 93.0),
                                     ("direction", "high"))),
            )
            name = "NBR burn-scar detector"
            desc = ("Burn-scar mapping from the absolute bi-temporal NBR "
                    "difference and an adaptive threshold.")
        targets = ("burn",)
    else:  # pragma: no cover
        raise ValueError(family)
    return _make_skill(name, "analysis", desc, ops, family, targets,
                       FAMILY_TASK_KIND[family], bands)


def _refinement_skill(family: str, variant: int) -> Skill:
    if family == "landcover":
        ops = (OpSpec("smooth", (("method", "majority"), ("radius", 1 + variant % 2))),)
        name = "majority-filter class cleanup"
        desc = "Majority filter that removes isolated land-cover pixels."
    elif family == "crop":
        ops = (OpSpec("remove_small", (("min_size", 10 + 8 * (variant % 3)),)),)
        name = "small-field removal"
        desc = "Remove small connected crop patches below a minimum field size."
    elif family in {"water", "flood"}:
        ops = (
            OpSpec("smooth", (("method", "majority"), ("radius", 1 + variant % 2))),
            OpSpec("remove_small", (("min_size", 8 + 8 * (variant % 2)),)),
        )
        name = "water-mask cleanup"
        desc = "Majority smoothing followed by removal of small water components."
    elif family in {"change", "urban"}:
        ops = ()
        name = "no cleanup"
        desc = ("Preserve the raw analysis map without spatial cleanup; "
                "appropriate when change patches are small and fragmented.")
    elif family == "burn":
        ops = (OpSpec("remove_small", (("min_size", 6 + 4 * (variant % 2)),)),)
        name = "burn-scar component filter"
        desc = "Remove isolated burn pixels while keeping compact scar patches."
    else:  # pragma: no cover
        raise ValueError(family)
    return _make_skill(name, "refinement", desc, ops, family, (FAMILY_TARGET[family],),
                       FAMILY_TASK_KIND[family], ("blue", "green", "red", "nir", "swir1", "swir2"))


def _make_skill(name: str, kind: str, description: str, ops: tuple[OpSpec, ...],
                family: str, targets: tuple[str, ...], input_kind: str,
                bands: tuple[str, ...]) -> Skill:
    digest = hashlib.sha1(
        repr((name, kind, targets, tuple(op.canonical() for op in ops))).encode()
    ).hexdigest()[:10]
    signature = hashlib.sha1(
        repr((kind, targets, tuple(op.canonical() for op in ops))).encode()
    ).hexdigest()[:16]
    return Skill(
        skill_id=f"{kind[:3]}-{family}-{digest}",
        name=name,
        kind=kind,
        description=description,
        ops=ops,
        source_family=family,
        targets=targets,
        input_kind=input_kind,
        required_bands=bands,
        params={"variant": 0},
        signature=signature,
    )


def expert_skills(spec: TaskSpec) -> tuple[Skill, Skill]:
    """Analysis and refinement halves of the reference solution."""
    index = int(hashlib.sha1(spec.task_id.encode()).hexdigest()[:8], 16)
    return _analysis_skill(spec.family, index), _refinement_skill(spec.family, index)


def expert_pipeline(spec: TaskSpec) -> Skill:
    """Complete reference recipe; this is the unit stored in the skill library."""
    analysis, refinement = expert_skills(spec)
    ops = analysis.ops + refinement.ops
    suffix = "" if refinement.ops else " without cleanup"
    description = analysis.description
    if refinement.ops:
        description = description + " " + refinement.description
    return _make_skill(analysis.name + suffix, "pipeline", description, ops,
                       spec.family, analysis.targets, analysis.input_kind,
                       analysis.required_bands)


def generic_analysis(spec: TaskSpec) -> Skill:
    """A domain-agnostic fallback used when no historical skill is applicable."""
    if spec.bitemporal:
        ops = (
            OpSpec("temporal_diff", (("index", "ndvi"), ("absolute", True))),
            OpSpec("threshold", (("method", "fixed"), ("value", 0.10), ("direction", "high"))),
        )
        desc = "Generic absolute NDVI difference with a fixed threshold."
    else:
        ops = (
            OpSpec("index", (("name", "ndvi"), ("date", "post"))),
            OpSpec("threshold", (("method", "fixed"), ("value", 0.20), ("direction", "high"))),
        )
        desc = "Generic NDVI vegetation mask with a fixed threshold."
    return _make_skill("generic NDVI baseline", "analysis", desc, ops, "generic",
                       (FAMILY_TARGET[spec.family],), spec.input_kind,
                       ("blue", "green", "red", "nir", "swir1", "swir2"))


def generic_refinement(spec: TaskSpec) -> Skill:
    return _make_skill("generic no-op refinement", "refinement",
                       "Leave the raw analysis map unchanged.",
                       (), "generic", (FAMILY_TARGET[spec.family],), spec.input_kind,
                       ("blue", "green", "red", "nir", "swir1", "swir2"))


def generic_pipeline(spec: TaskSpec) -> Skill:
    analysis = generic_analysis(spec)
    return _make_skill(analysis.name, "pipeline", analysis.description, analysis.ops,
                       "generic", analysis.targets, analysis.input_kind,
                       analysis.required_bands)


def build_library(specs: list[TaskSpec], include_generic: bool = True) -> list[Skill]:
    """Induce a deduplicated skill library from reference traces."""
    library: dict[str, Skill] = {}
    for spec in specs:
        skill = expert_pipeline(spec)
        existing = library.get(skill.signature)
        if existing is None:
            skill.stats.source_tasks = (spec.task_id,)
            library[skill.signature] = skill
        else:
            if spec.task_id not in existing.stats.source_tasks:
                existing.stats.source_tasks = existing.stats.source_tasks + (spec.task_id,)
    skills = list(library.values())
    if include_generic:
        # Generic skills are part of every retrieval pool but are not counted as
        # "historical skills" in library-size analyses.
        pass
    return skills


def success_threshold(spec: TaskSpec) -> float:
    return SUCCESS_THRESHOLD[spec.family]


def is_success(spec: TaskSpec, score: float) -> bool:
    return score >= success_threshold(spec)


def library_for_families(library: list[Skill], families: tuple[str, ...] | list[str]) -> list[Skill]:
    allowed = set(families)
    return [skill for skill in library if skill.source_family in allowed]
