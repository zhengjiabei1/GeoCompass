"""Shared data structures for the SkillEvo benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
CLASS_NAMES = ("water", "forest", "grass", "crop", "urban", "bare")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

# Target families.  ``targets`` is the semantic label the agent sees in the
# task description; it is intentionally coarser than the hidden ground truth.
FAMILIES = (
    "landcover",
    "crop",
    "water",
    "flood",
    "change",
    "urban",
    "burn",
)

FAMILY_TARGET = {
    "landcover": "landcover",
    "crop": "crop",
    "water": "water",
    "flood": "flood",
    "change": "change",
    "urban": "urban_change",
    "burn": "burn",
}

FAMILY_TASK_KIND = {
    "landcover": "single",
    "crop": "single",
    "water": "single",
    "flood": "single",
    "change": "bitemporal",
    "urban": "bitemporal",
    "burn": "bitemporal",
}

# Expected spectral evidence for a family.  This is domain knowledge that is
# available in the task description, not ground truth.
FAMILY_EVIDENCE = {
    "landcover": "multispectral",
    "crop": "vegetation_index",
    "water": "ndwi",
    "flood": "ndwi",
    "change": "ndvi_change",
    "urban": "ndbi_change",
    "burn": "nbr_change",
}


@dataclass(frozen=True)
class TaskSpec:
    """A task template with all information visible to the agent."""

    task_id: str
    family: str
    split: str
    seed: int
    region: str = "central"
    season: str = "summer"
    cloud: float = 0.0
    resolution: int = 10
    size: int = 32
    change_type: str = "mixed"
    n_classes: int = 6
    input_kind: str = ""
    targets: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        object.__setattr__(self, "input_kind", self.input_kind or FAMILY_TASK_KIND[self.family])
        if not self.targets:
            object.__setattr__(self, "targets", (FAMILY_TARGET[self.family],))

    @property
    def target(self) -> str:
        return self.targets[0]

    @property
    def bitemporal(self) -> bool:
        return self.input_kind == "bitemporal"

    def visible(self) -> dict[str, Any]:
        """Description passed to retrieval / feature extraction."""
        return {
            "task_id": self.task_id,
            "family": self.family,
            "target": self.target,
            "input_kind": self.input_kind,
            "bands": list(BANDS),
            "region": self.region,
            "season": self.season,
            "cloud": self.cloud,
            "resolution": self.resolution,
            "n_classes": self.n_classes,
            "evidence": FAMILY_EVIDENCE[self.family],
        }


@dataclass
class Scene:
    """Synthetic multi-band scene plus hidden ground truth."""

    size: int
    pre: dict[str, list[float]]
    post: dict[str, list[float]]
    classes_pre: list[int]
    classes_post: list[int]
    change_mask: list[int]
    valid: list[int]
    truth: list[int]
    truth_kind: str
    samples: list[int]
    sample_labels: list[int]
    change_type: str = "mixed"
    cloud_fraction: float = 0.0

    @property
    def n_pixels(self) -> int:
        return self.size * self.size


@dataclass
class TaskInstance:
    spec: TaskSpec
    scene: Scene
    expert_analysis: "Skill | None" = None
    expert_refinement: "Skill | None" = None


@dataclass(frozen=True)
class OpSpec:
    """One primitive operation in a skill macro."""

    op: str
    params: tuple[tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        for name, value in self.params:
            if name == key:
                return value
        return default

    def as_dict(self) -> dict[str, Any]:
        return {"op": self.op, **dict(self.params)}

    def canonical(self, adaptive: bool = True) -> tuple:
        """Stable representation; adaptive parameters are erased."""
        items = []
        for key, value in self.params:
            if adaptive and key in {"threshold", "value"}:
                value = "auto"
            if isinstance(value, list):
                value = tuple(value)
            items.append((key, value))
        return (self.op, tuple(items))


@dataclass
class SkillStats:
    uses: int = 0
    successes: int = 0
    utility_sum: float = 0.0
    utility_sq_sum: float = 0.0
    last_seen: int = -1
    source_tasks: tuple[str, ...] = ()

    def observe(self, success: bool, utility: float, step: int, task_id: str) -> None:
        self.uses += 1
        self.successes += int(success)
        self.utility_sum += utility
        self.utility_sq_sum += utility * utility
        self.last_seen = step
        if task_id not in self.source_tasks:
            self.source_tasks = self.source_tasks + (task_id,)

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    @property
    def mean_utility(self) -> float:
        return self.utility_sum / self.uses if self.uses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uses": self.uses,
            "successes": self.successes,
            "utility_sum": round(self.utility_sum, 6),
            "utility_sq_sum": round(self.utility_sq_sum, 6),
            "last_seen": self.last_seen,
            "source_tasks": list(self.source_tasks[-8:]),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SkillStats":
        return cls(
            uses=int(row.get("uses", 0)),
            successes=int(row.get("successes", 0)),
            utility_sum=float(row.get("utility_sum", 0.0)),
            utility_sq_sum=float(row.get("utility_sq_sum", 0.0)),
            last_seen=int(row.get("last_seen", -1)),
            source_tasks=tuple(row.get("source_tasks", ())),
        )


@dataclass
class Skill:
    """A reusable analysis (or refinement) macro induced from past tasks."""

    skill_id: str
    name: str
    kind: str  # "analysis" | "refinement"
    description: str
    ops: tuple[OpSpec, ...]
    source_family: str
    targets: tuple[str, ...]
    input_kind: str
    required_bands: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    stats: SkillStats = field(default_factory=SkillStats)
    signature: str = ""
    created_step: int = 0
    duplicate_of: str | None = None
    active: bool = True

    @property
    def adaptive(self) -> bool:
        return any(
            op.get("threshold") == "auto"
            or op.get("adaptive")
            or op.get("method") in {"otsu", "percentile", "mean_sd"}
            for op in self.ops
        )

    def canonical_ops(self) -> tuple:
        return tuple(op.canonical() for op in self.ops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "ops": [op.as_dict() for op in self.ops],
            "source_family": self.source_family,
            "targets": list(self.targets),
            "input_kind": self.input_kind,
            "required_bands": list(self.required_bands),
            "params": dict(self.params),
            "stats": self.stats.to_dict(),
            "signature": self.signature,
            "created_step": self.created_step,
            "duplicate_of": self.duplicate_of,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Skill":
        ops = tuple(
            OpSpec(item["op"], tuple((k, v) for k, v in item.items() if k != "op"))
            for item in row["ops"]
        )
        return cls(
            skill_id=row["skill_id"],
            name=row["name"],
            kind=row["kind"],
            description=row["description"],
            ops=ops,
            source_family=row["source_family"],
            targets=tuple(row.get("targets", ())),
            input_kind=row["input_kind"],
            required_bands=tuple(row.get("required_bands", ())),
            params=dict(row.get("params", {})),
            stats=SkillStats.from_dict(row.get("stats", {})),
            signature=row.get("signature", ""),
            created_step=int(row.get("created_step", 0)),
            duplicate_of=row.get("duplicate_of"),
            active=bool(row.get("active", True)),
        )


@dataclass
class Outcome:
    """Observed result of executing one skill on one task."""

    task_id: str
    task_family: str
    target: str
    skill_id: str
    kind: str
    success: bool
    metric: float
    baseline_metric: float
    utility: float
    step: int
    stage: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "target": self.target,
            "skill_id": self.skill_id,
            "kind": self.kind,
            "success": self.success,
            "metric": round(self.metric, 6),
            "baseline_metric": round(self.baseline_metric, 6),
            "utility": round(self.utility, 6),
            "step": self.step,
            "stage": self.stage,
            **self.extra,
        }
