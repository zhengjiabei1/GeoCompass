"""Candidate retrieval over the skill library.

The benchmark originally generated candidates with a hashed bag-of-words
embedding.  That is a convenient sanity check but a weak information-retrieval
baseline, and comparing a reranker against it overstates the value of the
reranker.  This module adds a real lexical ranker (BM25) and a hybrid that
fuses lexical evidence with the description embedding, so that every reported
reranking gain is measured against a competitive first stage.

BM25 is used in the Okapi formulation with the usual term-frequency saturation
and length normalisation; the identifier forms are documented in
``docs/RELATED_WORK.md`` and ``docs/BENCHMARK.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .features import HashedEmbedding, cosine
from .types import Skill, TaskSpec

_TOKEN = re.compile(r"[a-z0-9]+")

BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, and expand identifier forms."""
    return _TOKEN.findall(text.lower().replace("_", " ").replace("-", " "))


def skill_document(skill: Skill) -> str:
    """Indexable text for one skill.

    The document mixes the prose description with the machine-readable schema
    (targets, input kind, bands, primitive sequence).  Schema tokens are what
    make lexical matching competitive: two recipes can read differently yet
    declare the same mechanism.
    """
    parts = [
        skill.name,
        skill.description,
        skill.source_family,
        skill.input_kind,
        *skill.targets,
        *skill.required_bands,
    ]
    parts.extend(op.op for op in skill.ops)
    if skill.adaptive:
        parts.append("adaptive")
    return " ".join(str(part) for part in parts)


def task_query(task: TaskSpec) -> str:
    """Indexable text for one task, built only from information the agent sees."""
    visible = task.visible()
    parts = [
        visible["family"],
        visible["target"],
        visible["input_kind"],
        visible["evidence"],
        visible["region"],
        visible["season"],
        *visible["bands"],
    ]
    if visible["cloud"]:
        parts.append("cloud")
    return " ".join(str(part) for part in parts)


@dataclass
class BM25Index:
    """Inverted index over the active skills of a library."""

    skills: list[Skill]
    k1: float = BM25_K1
    b: float = BM25_B

    def __post_init__(self) -> None:
        self.documents = [tokenize(skill_document(skill)) for skill in self.skills]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for document in self.documents:
            counts: dict[str, int] = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            self.term_frequencies.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        tokens = tokenize(query)
        if not tokens or not self.documents:
            return [0.0 for _ in self.skills]
        unique = list(dict.fromkeys(tokens))
        out = []
        for position, counts in enumerate(self.term_frequencies):
            length = self.lengths[position] or 1
            total = 0.0
            for token in unique:
                frequency = counts.get(token)
                if not frequency:
                    continue
                saturation = frequency * (self.k1 + 1.0)
                saturation /= frequency + self.k1 * (1.0 - self.b + self.b * length / (self.average_length or 1.0))
                total += self.idf.get(token, 0.0) * saturation
            out.append(total)
        return out


class BM25Retriever:
    """Lexical first stage, returning ``(skill, score)`` pairs."""

    name = "bm25"

    def __init__(self, library: list[Skill]):
        self.library = [skill for skill in library]
        self.index = BM25Index([skill for skill in self.library if skill.active])

    def candidates(self, task: TaskSpec, k: int = 8,
                   exclude: set[str] | None = None) -> list[tuple[Skill, float]]:
        scores = self.index.scores(task_query(task))
        scored = [
            (skill, scores[position])
            for position, skill in enumerate(self.index.skills)
            if not exclude or skill.skill_id not in exclude
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


class HybridRetriever:
    """Reciprocal-rank fusion of BM25 and the description embedding.

    Rank fusion is used rather than score averaging because the two scorers
    live on different scales and their distributions shift between libraries.
    """

    name = "hybrid"

    def __init__(self, library: list[Skill], embedding: HashedEmbedding | None = None,
                 weight: float = 60.0):
        self.library = list(library)
        self.embedding = embedding or HashedEmbedding()
        self.weight = weight
        self.lexical = BM25Retriever(self.library)

    def candidates(self, task: TaskSpec, k: int = 8,
                   exclude: set[str] | None = None) -> list[tuple[Skill, float]]:
        depth = max(k, 8)
        lexical = self.lexical.candidates(task, k=len(self.library), exclude=exclude)
        query = self.embedding.task(task)
        semantic = [
            (skill, cosine(query, self.embedding.skill(skill)))
            for skill in self.library
            if skill.active and (not exclude or skill.skill_id not in exclude)
        ]
        semantic.sort(key=lambda pair: pair[1], reverse=True)
        fused: dict[str, float] = {}
        for rank, (skill, _) in enumerate(lexical[:depth], start=1):
            fused[skill.skill_id] = fused.get(skill.skill_id, 0.0) + 1.0 / (self.weight + rank)
        for rank, (skill, _) in enumerate(semantic[:depth], start=1):
            fused[skill.skill_id] = fused.get(skill.skill_id, 0.0) + 1.0 / (self.weight + rank)
        by_id = {skill.skill_id: skill for skill in self.library}
        scored = [(by_id[skill_id], score) for skill_id, score in fused.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def build_retriever(kind: str, library: list[Skill]) -> object:
    """Factory used by the experiment runner and the ablations."""
    if kind == "bm25":
        return BM25Retriever(library)
    if kind == "hybrid":
        return HybridRetriever(library)
    raise ValueError(f"unknown retriever: {kind}")
