"""Tests for BM25 and hybrid candidate retrieval."""
from __future__ import annotations

import unittest

from geocompass.retrieval import BM25Index, BM25Retriever, HybridRetriever, tokenize
from geocompass.tasks import build_library, build_task_specs


class TokenizerTests(unittest.TestCase):
    def test_identifier_splitting(self) -> None:
        self.assertEqual(tokenize("ndvi_change"), ["ndvi", "change"])
        self.assertEqual(tokenize("MNDWI, Otsu"), ["mndwi", "otsu"])

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize(""), [])


class BM25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        splits = build_task_specs(0, {"source_train": 10, "source_val": 4,
                                      "target_dev": 8, "target_test": 6, "stream": 8})
        cls.library = build_library(splits["source_train"])
        cls.tasks = splits["target_test"]

    def test_index_statistics(self) -> None:
        index = BM25Index(self.library)
        self.assertEqual(len(index.skills), len(self.library))
        self.assertTrue(index.average_length > 0)
        self.assertTrue(all(value > 0 for value in index.idf.values()))

    def test_exact_schema_match_ranks_first(self) -> None:
        retriever = BM25Retriever(self.library)
        task = next(task for task in self.tasks if task.family == "water")
        candidates = retriever.candidates(task, k=len(self.library))
        self.assertTrue(candidates)
        top = candidates[0][0]
        self.assertIn(task.target, top.targets)
        self.assertEqual(top.input_kind, task.input_kind)

    def test_scores_are_sorted_and_truncated(self) -> None:
        retriever = BM25Retriever(self.library)
        candidates = retriever.candidates(self.tasks[0], k=3)
        self.assertLessEqual(len(candidates), 3)
        scores = [score for _, score in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_exclude_is_respected(self) -> None:
        retriever = BM25Retriever(self.library)
        excluded = {self.library[0].skill_id}
        candidates = retriever.candidates(self.tasks[0], k=len(self.library), exclude=excluded)
        self.assertNotIn(self.library[0].skill_id, {skill.skill_id for skill, _ in candidates})

    def test_hybrid_returns_unique_skills(self) -> None:
        retriever = HybridRetriever(self.library)
        candidates = retriever.candidates(self.tasks[0], k=8)
        identifiers = [skill.skill_id for skill, _ in candidates]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_hybrid_matches_lexical_top(self) -> None:
        lexical = BM25Retriever(self.library).candidates(self.tasks[0], k=1)[0][0]
        hybrid = [skill for skill, _ in HybridRetriever(self.library).candidates(self.tasks[0], k=3)]
        self.assertIn(lexical.skill_id, {skill.skill_id for skill in hybrid})


if __name__ == "__main__":
    unittest.main()
