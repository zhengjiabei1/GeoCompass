"""Fast smoke tests for the GeoCompass benchmark."""
from __future__ import annotations

import unittest

from geocompass.raster import generate_scene
from geocompass.tasks import FAMILIES, build_task_specs


class TestTaskFamilies(unittest.TestCase):
    def test_all_families_are_present(self) -> None:
        splits = build_task_specs(
            seed=0,
            per_split={
                "source_train": 1,
                "source_val": 1,
                "target_dev": 1,
                "target_test": 1,
                "stream": 1,
            },
        )
        seen = {task.family for tasks in splits.values() for task in tasks}
        self.assertEqual(seen, set(FAMILIES))

    def test_scene_shapes_are_consistent(self) -> None:
        splits = build_task_specs(seed=0, per_split={"target_test": 1})
        task = splits["target_test"][0]
        scene = generate_scene(task)
        self.assertEqual(len(scene.pre["red"]), task.size * task.size)
        self.assertEqual(len(scene.classes_pre), task.size * task.size)
        self.assertEqual(len(scene.truth), task.size * task.size)


if __name__ == "__main__":
    unittest.main()
