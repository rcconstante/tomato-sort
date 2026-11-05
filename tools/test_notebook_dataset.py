"""Regression checks for the generated notebook's dataset-path cell."""
import json
import tempfile
import unittest
from pathlib import Path

import yaml


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "TomatoSort_AI_Training.ipynb"
CELL = next(
    cell["source"] for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    if cell["cell_type"] == "code" and cell["source"].startswith("# Normalise data.yaml")
)


class DatasetPathsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"names": ["ripe", "unripe"], "train": "train/images"}
        self.image("train/images")

    def image(self, directory):
        target = self.root / directory
        target.mkdir(parents=True, exist_ok=True)
        (target / "sample.jpg").touch()
        return target.resolve()

    def run_cell(self):
        data_yaml = self.root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        scope = {"Path": Path, "yaml": yaml, "DATA_DIR": self.root, "DATA_YAML": data_yaml}
        exec(compile(CELL, str(NOTEBOOK), "exec"), scope)
        return scope["SPLITS"]

    def test_discovers_valid_without_yaml_val_entry(self):
        expected = self.image("valid/images")
        self.assertEqual(self.run_cell()["val"], expected)

    def test_recovers_stale_absolute_export_path(self):
        self.config["val"] = "/old-export/valid/images"
        expected = self.image("valid/images")
        self.assertEqual(self.run_cell()["val"], expected)

    def test_honors_yaml_root_and_validation_alias(self):
        self.config.update(path="nested", train="images/train", validation="images/validation")
        self.image("nested/images/train")
        expected = self.image("nested/images/validation")
        self.assertEqual(self.run_cell()["val"], expected)

    def test_too_few_source_groups_fails_without_rewriting_yaml(self):
        (self.root / "valid/images").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "at least two source-image groups"):
            self.run_cell()
        self.assertEqual(yaml.safe_load((self.root / "data.yaml").read_text()), self.config)

    def add_train_image(self, name):
        image = self.root / "train/images" / name
        image.write_bytes(name.encode())
        labels = self.root / "train/labels"
        labels.mkdir(exist_ok=True)
        (labels / f"{image.stem}.txt").write_text("0 0.5 0.5 0.2 0.2")
        return image

    def test_training_only_export_creates_disjoint_holdout_with_labels(self):
        self.add_train_image("sample.jpg")
        for index in range(10):
            self.add_train_image(f"tomato{index}.rf.first.jpg")
            self.add_train_image(f"tomato{index}.rf.second.jpg")
        original = set((self.root / "train/images").iterdir())
        splits = self.run_cell()
        train = {p.name for p in splits["train"].iterdir()}
        val = {p.name for p in splits["val"].iterdir()}
        self.assertTrue(train and val)
        self.assertFalse(train & val)
        self.assertEqual(train | val, {p.name for p in original})
        self.assertEqual(set((self.root / "train/images").iterdir()), original)
        train_groups = {Path(p).stem.split(".rf.")[0] for p in train}
        val_groups = {Path(p).stem.split(".rf.")[0] for p in val}
        self.assertFalse(train_groups & val_groups)
        for directory in splits.values():
            for image in directory.iterdir():
                self.assertTrue((directory.parent / "labels" / f"{image.stem}.txt").is_file())
        # Re-executing the cell uses the saved split without creating another copy.
        saved = yaml.safe_load((self.root / "data.yaml").read_text())
        self.config = saved
        self.assertEqual(self.run_cell(), splits)
        self.assertEqual(len(list(self.root.glob("holdout-seed42-*"))), 1)

    def test_holdout_rejects_missing_labels_before_copying(self):
        self.add_train_image("second.jpg")
        with self.assertRaisesRegex(RuntimeError, "images lack labels"):
            self.run_cell()
        self.assertEqual(list(self.root.glob("holdout-seed42-*")), [])

    def test_same_train_validation_directory_is_rejected(self):
        self.config["val"] = "train/images"
        with self.assertRaisesRegex(RuntimeError, "same directory"):
            self.run_cell()


if __name__ == "__main__":
    unittest.main()
