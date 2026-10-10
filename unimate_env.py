"""Finding UniMate, and running its code from ComfyUI's Python.

Two environments are in play and they are not interchangeable:

* **ComfyUI's venv** runs inference. It already carries everything UniMate's
  sampler needs (torch, transformers, numpy, scipy, einops, loguru,
  accelerate, sentencepiece); the only gaps are ``tyro``, used solely by the
  CLI's argument parser, and the ``Motion`` package, which is pure Python and
  therefore costs nothing but a ``sys.path`` entry. So sampling runs here, on
  the GPU ComfyUI already owns.
* **UniMate's own venv** runs `rig_preprocess`, because that needs ``bpy`` --
  a ~1 GB wheel pinned to a Blender version. Installing it into ComfyUI's venv
  to prepare a rig once per asset would be a poor trade, so preparation is a
  subprocess against UniMate's interpreter instead. Same pattern as
  ComfyUI-SkinToken's Blender bridge.

Both locations are resolved once, here, with an override so nobody has to edit
code: set ``UNIMATE_REPO`` (and optionally ``UNIMATE_PYTHON``) in the
environment, or drop a ``unimate_path.txt`` beside this file holding the repo
path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Where a UniMate checkout usually sits, tried in order. The override comes
# first so a configured path always wins.
_REPO_CANDIDATES = (
    "C:/Git/UniMate",
    "~/UniMate",
    "/opt/UniMate",
)


class UniMateNotFound(RuntimeError):
    """The UniMate checkout could not be located."""


def _looks_like_repo(path: Path) -> bool:
    """A real checkout, not just a folder of that name."""
    return (path / "unimate" / "inference" / "sample.py").is_file()


def find_repo() -> Path:
    """The UniMate checkout. Raises :class:`UniMateNotFound` with what to do."""
    candidates = []
    env = os.environ.get("UNIMATE_REPO")
    if env:
        candidates.append(Path(env).expanduser())
    marker = HERE / "unimate_path.txt"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if text:
            candidates.append(Path(text).expanduser())
    candidates += [Path(p).expanduser() for p in _REPO_CANDIDATES]

    for path in candidates:
        if _looks_like_repo(path):
            return path.resolve()

    raise UniMateNotFound(
        "Could not find the UniMate checkout. Clone "
        "https://github.com/Friedrich-M/UniMate and then either set the "
        "UNIMATE_REPO environment variable to it, or write its path into "
        f"{HERE / 'unimate_path.txt'}.\nLooked in: "
        + ", ".join(str(c) for c in candidates)
    )


def ensure_importable() -> Path:
    """Put UniMate (and its bundled ``Motion``) on ``sys.path``; return the repo.

    ``Motion`` is imported by UniMate as top-level ``Animation`` /
    ``Quaternions``, not as ``Motion.Animation``, so the subdirectory itself
    goes on the path rather than its parent.
    """
    repo = find_repo()
    for entry in (repo, repo / "Motion"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def repo_python() -> str:
    """Interpreter for the bpy-dependent preprocessing step.

    UniMate's own venv when it has one (that is where ``bpy`` lives), else
    whatever ``UNIMATE_PYTHON`` names, else this process's interpreter -- which
    works only if ``bpy`` happens to be installed here.
    """
    env = os.environ.get("UNIMATE_PYTHON")
    if env and Path(env).is_file():
        return env
    repo = find_repo()
    for rel in ("venv/Scripts/python.exe", "venv/bin/python",
                ".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = repo / rel
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def has_bpy(python: str | None = None) -> bool:
    """Whether *python* can import bpy -- what preparation actually needs."""
    import subprocess
    exe = python or repo_python()
    try:
        done = subprocess.run([exe, "-c", "import bpy"], capture_output=True, timeout=120)
        return done.returncode == 0
    except Exception:
        return False
