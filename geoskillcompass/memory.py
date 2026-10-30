"""Tiered experience memory with consolidation, deduplication, and pruning."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .features import MemoryStats
from .types import Outcome

VALID_MODES = ("none", "flat", "tiered", "tiered_dedup", "full")


@dataclass
class MemoryRecord:
    record_id: int
    step: int
    skill_id: str
    signature: str
    task_id: str
    task_family: str
    target: str
    kind: str
    success: bool
    utility: float
    stage: int = 0

    @classmethod
    def from_outcome(cls, record_id: int, outcome: Outcome) -> "MemoryRecord":
        return cls(
            record_id=record_id,
            step=outcome.step,
            skill_id=outcome.skill_id,
            signature=str(outcome.extra.get("signature", outcome.skill_id)),
            task_id=outcome.task_id,
            task_family=outcome.task_family,
            target=outcome.target,
            kind=outcome.kind,
            success=outcome.success,
            utility=outcome.utility,
            stage=outcome.stage,
        )

    def bucket(self) -> tuple:
        return (
            self.skill_id,
            self.task_family,
            self.target,
            bool(self.success),
            round(self.utility, 1),
        )


class ExperienceMemory:
    """Short-term cache plus consolidated long-term experience.

    ``flat`` keeps every episode, ``tiered`` adds a short-term window and a
    bounded long-term store, ``tiered_dedup`` collapses duplicate outcome
    buckets, and ``full`` additionally prunes low-utility records.
    """

    def __init__(self, mode: str = "full", short_window: int = 24,
                 capacity: int = 140, prune_to: int = 90):
        if mode not in VALID_MODES:
            raise ValueError(f"unknown memory mode: {mode}")
        self.mode = mode
        self.short_window = short_window
        self.capacity = capacity
        self.prune_to = min(prune_to, capacity)
        self.short_term: deque[MemoryRecord] = deque(maxlen=short_window)
        self.records: list[MemoryRecord] = []
        self.seen = 0
        self.duplicates = 0
        self._counter = 0
        self.consolidations = 0
        self._skill_seen: dict[str, int] = {}

    # ------------------------------------------------------------------ ingest
    def ingest(self, outcome: Outcome) -> None:
        self.seen += 1
        if self.mode == "none":
            return
        self._counter += 1
        record = MemoryRecord.from_outcome(self._counter, outcome)
        self.short_term.append(record)
        self._skill_seen[record.skill_id] = self._skill_seen.get(record.skill_id, 0) + 1

        if self.mode == "flat":
            self.records.append(record)
            return

        key = record.bucket()
        existing = {row.bucket() for row in self.records}
        if self.mode in {"tiered_dedup", "full"} and key in existing:
            self.duplicates += 1
            # Keep the freshest representative instead of counting the same
            # outcome repeatedly.
            for index, row in enumerate(self.records):
                if row.bucket() == key:
                    self.records[index] = record
                    break
        else:
            self.records.append(record)

        if len(self.records) > self.capacity:
            self._consolidate()

    def _consolidate(self) -> None:
        self.consolidations += 1
        if self.mode == "full":
            # Compression: keep high-utility, recent, and repeated evidence.
            counts: dict[str, int] = {}
            for record in self.records:
                counts[record.signature] = counts.get(record.signature, 0) + 1
            self.records.sort(
                key=lambda row: (
                    counts.get(row.signature, 1),
                    row.utility,
                    row.step,
                ),
                reverse=True,
            )
            self.records = self.records[: self.prune_to]
        else:
            self.records = self.records[-self.prune_to:]

    # ------------------------------------------------------------------- stats
    def stats(self, skill_id: str, task_family: str | None = None) -> MemoryStats:
        if self.mode == "none":
            return MemoryStats()
        rows = [row for row in self.records if row.skill_id == skill_id]
        recent = [row for row in self.short_term if row.skill_id == skill_id]
        family_rows = [row for row in rows if task_family is not None and row.task_family == task_family]
        total = self._skill_seen.get(skill_id, 0)
        stats = MemoryStats(
            uses=len(rows),
            successes=sum(int(row.success) for row in rows),
            utility_sum=sum(row.utility for row in rows),
            utility_sq_sum=sum(row.utility * row.utility for row in rows),
            last_seen=max((row.step for row in rows), default=-1),
            duplicates=max(0, total - len(rows)),
            coverage=len({row.task_family for row in rows}),
            family_uses=len(family_rows),
            family_successes=sum(int(row.success) for row in family_rows),
            retained=len(rows),
            total_seen=total,
            recent_uses=len(recent),
            recent_successes=sum(int(row.success) for row in recent),
        )
        return stats

    def stats_by_signature(self, signature: str) -> list[MemoryStats]:
        rows = [row for row in self.records if row.signature == signature]
        if not rows:
            return []
        return [
            MemoryStats(
                uses=len(rows),
                successes=sum(int(row.success) for row in rows),
                utility_sum=sum(row.utility for row in rows),
                utility_sq_sum=sum(row.utility * row.utility for row in rows),
                last_seen=max(row.step for row in rows),
                retained=len(rows),
            )
        ]

    # ------------------------------------------------------------------ report
    def snapshot(self) -> dict[str, float]:
        chars = sum(len(row.skill_id) + len(row.task_id) + 48 for row in self.records)
        active_skills = len({row.skill_id for row in self.records})
        duplicate_ratio = self.duplicates / max(1, self.seen)
        return {
            "mode": self.mode,
            "seen": self.seen,
            "retained": len(self.records),
            "short_term": len(self.short_term),
            "characters": chars,
            "active_skills": active_skills,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "consolidations": self.consolidations,
            "compression_ratio": round(len(self.records) / max(1, self.seen), 4),
            "knowledge_density": round(active_skills / max(1, len(self.records)), 4),
        }

    def coverage(self) -> float:
        if not self.records:
            return 0.0
        return len({row.task_family for row in self.records}) / 7.0

    def utility_mass(self) -> float:
        return sum(max(0.0, row.utility) for row in self.records)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "short_window": self.short_window,
            "capacity": self.capacity,
            "prune_to": self.prune_to,
            "records": [
                {
                    "record_id": row.record_id,
                    "step": row.step,
                    "skill_id": row.skill_id,
                    "signature": row.signature,
                    "task_id": row.task_id,
                    "task_family": row.task_family,
                    "target": row.target,
                    "kind": row.kind,
                    "success": row.success,
                    "utility": row.utility,
                    "stage": row.stage,
                }
                for row in self.records
            ],
            "seen": self.seen,
            "duplicates": self.duplicates,
            "snapshot": self.snapshot(),
        }


def beta_mean(successes: int, uses: int, prior: float = 0.5, strength: float = 2.0) -> float:
    return (successes + prior * strength) / (uses + strength)


def beta_variance(successes: int, uses: int, prior: float = 0.5, strength: float = 2.0) -> float:
    alpha = successes + prior * strength
    beta = (uses - successes) + (1.0 - prior) * strength
    return alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))


def entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log(probability) + (1 - probability) * math.log(1 - probability))
