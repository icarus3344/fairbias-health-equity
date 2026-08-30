"""Clean-checkout smoke tests.

A previous gate failed because HEAD committed `fairbias.evaluator` while the
modules it imports (`fairbias.models`, `fairbias.data`, `fairbias/__init__.py`)
remained untracked workspace files: the workspace ran fine, but a clean
checkout raised ``ModuleNotFoundError``.  These tests pin the invariant that
the Git HEAD tree alone is sufficient to import the full fairbias runtime.
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every module required at import time by fairbias.evaluator / fairbias.pipeline
_REQUIRED_FAIRBIAS_MODULES = {
    "__init__",
    "bias_metric",
    "config",
    "data",
    "enhancement",
    "evaluator",
    "mitigation",
    "models",
    "pipeline",
    "transform",
}


def _git(args, check=True):
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT)] + args,
        capture_output=True,
        text=True,
        check=check,
    )


@unittest.skipUnless(
    (_REPO_ROOT / ".git").exists(), "not a Git checkout"
)
class TestCleanCheckout(unittest.TestCase):
    """The committed HEAD tree must be self-sufficient for fairbias imports."""

    def test_head_tree_contains_all_fairbias_runtime_modules(self):
        """Every runtime module must be TRACKED (present in the HEAD tree).

        Workspace-only files cannot satisfy an import from a clean checkout.
        """
        result = _git(["ls-tree", "-r", "HEAD", "--name-only", "src/fairbias"])
        tracked = {
            pathlib.Path(line).stem
            for line in result.stdout.splitlines()
            if line.endswith(".py")
        }
        missing = _REQUIRED_FAIRBIAS_MODULES - tracked
        self.assertEqual(
            missing, set(),
            msg=(
                "src/fairbias modules missing from the Git HEAD tree (untracked "
                f"runtime dependencies break clean checkouts): {sorted(missing)}"
            ),
        )

    def test_golden_fixture_files_are_tracked(self):
        """Round-4.1 P0 regression guard: test fixtures must be TRACKED.

        The round-4 commit tracked ``PROVENANCE.json`` but left
        ``distance_matrix_step_0.csv`` untracked — the MDS golden tests
        passed on the workspace (where the file happened to exist) and
        failed with FileNotFoundError on a clean checkout.  Files under
        ``tests/fixtures/`` that exist in the workspace MUST therefore be
        tracked in the index/HEAD.
        """
        fixtures_dir = _REPO_ROOT / "tests" / "fixtures"
        result = _git(["ls-files", "--", "tests/fixtures"])
        tracked = set(result.stdout.splitlines())
        on_disk = {
            str(p.relative_to(_REPO_ROOT))
            for p in fixtures_dir.rglob("*")
            if p.is_file()
        }
        untracked = on_disk - tracked
        self.assertEqual(
            untracked, set(),
            msg=(
                "test fixture files present in the workspace but NOT tracked "
                f"in Git (clean checkouts will fail): {sorted(untracked)}"
            ),
        )

    def test_import_fairbias_from_head_archive(self):
        """Import the full runtime from an archive of HEAD (no workspace files).

        This reproduces exactly what a fresh ``git clone && git checkout HEAD``
        would see: only committed content, none of the local modifications or
        untracked files that can mask a broken commit.
        """
        with tempfile.TemporaryDirectory(prefix="fairbias_head_") as tmp:
            archive = pathlib.Path(tmp)
            # Export ONLY the committed tree (git archive reads HEAD, not the
            # working directory)
            proc = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "archive", "HEAD", "src"],
                capture_output=True,
                check=True,
            )
            untar = subprocess.run(
                ["tar", "-x", "-C", str(archive)],
                input=proc.stdout,
                capture_output=True,
                check=True,
            )
            self.assertEqual(untar.returncode, 0)

            # Import inside a subprocess with PYTHONPATH pointing at the
            # extracted HEAD copy (never the workspace src/)
            code = (
                "import fairbias.config, fairbias.data, fairbias.models, "
                "fairbias.evaluator, fairbias.transform, fairbias.mitigation, "
                "fairbias.enhancement, fairbias.pipeline; "
                "print('clean-checkout imports OK')"
            )
            env_path = str(archive / "src")
            import os

            env = dict(os.environ)
            env["PYTHONPATH"] = env_path
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                cwd=tmp,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=(
                    "Importing fairbias from the clean HEAD archive failed "
                    f"(a fresh checkout would break the same way):\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
