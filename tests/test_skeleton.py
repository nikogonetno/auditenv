"""Checkpoint 0 skeleton: supported interpreter, planned modules import, no runtime dependencies."""
import importlib
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = (
    "auditenv",
    "auditenv.world",
    "auditenv.inject",
    "auditenv.transform",
    "auditenv.env",
    "auditenv.grader",
    "auditenv.server",
    "policies",
)


class Skeleton(unittest.TestCase):
    def test_interpreter_is_supported(self):
        self.assertGreaterEqual(sys.version_info[:2], (3, 11))

    def test_planned_modules_import(self):
        for name in MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_no_runtime_dependencies(self):
        with open(ROOT / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)["project"]
        self.assertEqual(project["dependencies"], [], "verification must run offline")

    def test_sources_carry_the_mit_license(self):
        """Case §09: sources with a license and pinned dependencies."""
        with open(ROOT / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)["project"]
        self.assertEqual((project["license"], project["license-files"]), ("MIT", ["LICENSE"]))
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("MIT License\n"))
        self.assertIn("Permission is hereby granted, free of charge", text)
        self.assertIn("FROM python:3.11.11-slim-bookworm", (ROOT / "Dockerfile").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
