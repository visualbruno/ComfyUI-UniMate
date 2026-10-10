"""UniMate nodes: text-driven motion for any rigged skeleton.

The graph is five nodes, in the order the work actually happens:

    Model Loader ─┐
    Mesh Loader ──> Mesh Preparation ──> Mesh Sampler ──> Mesh Export

**Preparation is the slow, cacheable step** -- it runs Blender -- so it is its
own node and its results are cached by file content. Sampling is the fast part
you iterate on (prompt, cfg, seed), and export turns features into a playable
GLB.

Two deliberate choices:

* **Sampling shells out to UniMate's own CLI**, in ComfyUI's interpreter (which
  already has torch and everything else it needs). Their inference internals
  are moving fast -- `sample.py` grew 838 lines in a single release -- while
  the CLI surface is stable, so this keeps working across upstream churn. The
  cost is that weights load per run; for a 74M model that is seconds.
* **Preparation shells out to UniMate's venv**, because it needs `bpy`.

LICENCE: the UniMate checkpoints are CC BY-NC 4.0 and their model card states
that no assurance is given that generated motions are free of third-party
rights. These nodes are tooling; what you may do with the output is governed by
that licence. See the README.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import folder_paths

try:  # ComfyUI's 3D file type (Preview 3D input); absent in older ComfyUI versions
    from comfy_api.latest import Types

    File3D = Types.File3D
except (ImportError, AttributeError):
    File3D = None

from .unimate_env import UniMateNotFound, ensure_importable, find_repo, repo_python
from .glb_anim import motion_to_glb

CATEGORY = "3D/UniMate"

# A run directory is one of these laid out as the checkpoints ship:
# config.json + dataset_stats.npy + checkpoints/checkpoint_step_*.pt
_RUN_MARKERS = ("config.json", "dataset_stats.npy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    return Path(folder_paths.models_dir) / "unimate"


def _list_runs() -> list[str]:
    """Checkpoint run directories under models/unimate, newest first."""
    root = _models_dir()
    if not root.is_dir():
        return []
    runs = [d for d in root.iterdir()
            if d.is_dir() and all((d / m).is_file() for m in _RUN_MARKERS)]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def _list_rigged_meshes() -> list[str]:
    """Rigged meshes in ComfyUI's input directory (recursively)."""
    root = Path(folder_paths.get_input_directory())
    if not root.is_dir():
        return []
    out = []
    for ext in ("glb", "gltf", "fbx"):
        out += [str(p.relative_to(root)).replace("\\", "/")
                for p in root.rglob(f"*.{ext}")]
    return sorted(out)


def _digest(path: Path, extra: str = "") -> str:
    """Content hash of a file, so preparation is cached per actual mesh."""
    h = hashlib.sha1()
    h.update(extra.encode("utf-8"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _run(cmd: list[str], cwd: Path, what: str, timeout: int = 3600,
         env: dict | None = None) -> str:
    """Run a subprocess, raising with its tail if it fails."""
    done = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace",
                          env=env)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-25:])
        raise RuntimeError(f"{what} failed (exit {done.returncode}):\n{tail}")
    return done.stdout or ""


def _can_import(python: str, module: str) -> bool:
    """Whether *python* can import *module* -- cached, since it spawns a process."""
    key = (python, module)
    if key not in _IMPORT_CACHE:
        try:
            done = subprocess.run([python, "-c", f"import {module}"],
                                  capture_output=True, timeout=120)
            _IMPORT_CACHE[key] = done.returncode == 0
        except Exception:
            _IMPORT_CACHE[key] = False
    return _IMPORT_CACHE[key]


_IMPORT_CACHE: dict[tuple[str, str], bool] = {}


def _load_cond(asset_dir: Path) -> tuple[str, dict]:
    """The single cond entry of a rig_preprocess asset directory."""
    cond_path = asset_dir / "cond.npy"
    if not cond_path.is_file():
        raise RuntimeError(f"{asset_dir} has no cond.npy -- it is not a prepared asset")
    data = np.load(cond_path, allow_pickle=True).item()
    if "parents" in data:            # a bare entry rather than {name: entry}
        return asset_dir.name, data
    if len(data) != 1:
        raise RuntimeError(f"{cond_path} holds {len(data)} skeletons; expected one")
    name = next(iter(data))
    return name, data[name]


def _canonical_glb(asset_dir: Path, name: str) -> Path:
    """The canonical rest-pose GLB rig_preprocess bakes beside the cond."""
    direct = asset_dir / f"{name}.glb"
    if direct.is_file():
        return direct
    globbed = [p for p in asset_dir.glob("*.glb")]
    if len(globbed) == 1:
        return globbed[0]
    raise RuntimeError(
        f"No canonical GLB in {asset_dir}. rig_preprocess writes one only when the "
        f"source had a skinned mesh; a skeleton-only rig can be sampled but not exported.")


# ---------------------------------------------------------------------------
# 1 - Model Loader
# ---------------------------------------------------------------------------

class UniMateModelLoader:
    """Pick a checkpoint run and check it is complete before anything else runs."""

    @classmethod
    def INPUT_TYPES(cls):
        runs = _list_runs()
        return {
            "required": {
                "run": (runs or ["<put a run in models/unimate>"],),
                "checkpoint": (["latest", "earliest"], {"default": "latest"}),
            },
            "optional": {
                "run_path": ("STRING", {"default": "", "multiline": False,
                                        "tooltip": "Absolute path to a run directory; "
                                                   "overrides the dropdown."}),
            },
        }

    RETURN_TYPES = ("UNIMATE_MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Selects a UniMate checkpoint run (config.json + dataset_stats.npy "
                   "+ checkpoints/). Reports the joint and frame limits the rest of "
                   "the graph must respect.")

    def load(self, run: str, checkpoint: str, run_path: str = ""):
        exp_dir = Path(run_path).expanduser() if run_path.strip() else _models_dir() / run
        if not exp_dir.is_dir():
            raise RuntimeError(
                f"{exp_dir} is not a directory. Download a run from "
                f"https://huggingface.co/Linzhan/UniMate into {_models_dir()} "
                f"(keeping its config.json, dataset_stats.npy and checkpoints/).")
        for marker in _RUN_MARKERS:
            if not (exp_dir / marker).is_file():
                raise RuntimeError(f"{exp_dir} has no {marker}: incomplete checkpoint run.")

        ckpts = sorted((exp_dir / "checkpoints").glob("checkpoint_step_*.pt"))
        if not ckpts:
            raise RuntimeError(f"No checkpoint_step_*.pt in {exp_dir / 'checkpoints'}")

        def step_of(p: Path) -> int:
            digits = "".join(c for c in p.stem.split("_")[-1] if c.isdigit())
            return int(digits or 0)

        ckpts.sort(key=step_of)
        chosen = ckpts[-1] if checkpoint == "latest" else ckpts[0]

        cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
        ds, model = cfg.get("dataset", {}), cfg.get("model", {})
        handle = {
            "exp_dir": str(exp_dir),
            "checkpoint": str(chosen),
            "max_joints": int(ds.get("max_joints", 0)),
            "max_depth": int(ds.get("max_depth", 0)),
            "frames": int(ds.get("max_motion_length", 60)),
            "cfg_scale": float(cfg.get("sampling", {}).get("cfg_scale", 3.0)),
            "attention": model.get("attention", "?"),
            "text_cond": model.get("text_cond", "?"),
        }
        info = (f"{exp_dir.name} | step {step_of(chosen)} | {handle['attention']}+"
                f"{handle['text_cond']} | max {handle['max_joints']} joints, depth "
                f"{handle['max_depth']} | {handle['frames']} frames")
        return (handle, info)


# ---------------------------------------------------------------------------
# 2 - Mesh Loader
# ---------------------------------------------------------------------------

class UniMateMeshLoader:
    """Pick the rigged mesh to animate.

    Deliberately a path, not a TRIMESH: a trimesh carries geometry only, and
    everything here depends on the skeleton. The STRING this returns is the
    same shape as ComfyUI-SkinToken's ``rigged_path`` output, so a rig produced
    there can be wired straight in.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_rigged_meshes()
        return {
            "required": {
                "mesh": (files or ["<put a rigged .glb in ComfyUI/input>"],),
            },
            "optional": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Absolute path to a rigged "
                                                    "GLB/GLTF/FBX; overrides the dropdown."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("rigged_mesh", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Chooses a rigged GLB/GLTF/FBX. Accepts SkinToken's rigged_path."

    def load(self, mesh: str, mesh_path: str = ""):
        path = (Path(mesh_path).expanduser() if mesh_path.strip()
                else Path(folder_paths.get_input_directory()) / mesh)
        if not path.is_file():
            raise RuntimeError(f"No such mesh: {path}")
        size = path.stat().st_size / 1e6
        return (str(path), f"{path.name} ({size:.1f} MB)")


# ---------------------------------------------------------------------------
# 3 - Mesh Preparation
# ---------------------------------------------------------------------------

class UniMateMeshPreparation:
    """Turn a rigged mesh into an asset the model can be conditioned on.

    This is UniMate's own ``rig_preprocess``: it exports the skeleton, labels
    every joint against the model's anatomical vocabulary, picks the facing
    pair, canonicalises the rest pose and bakes a canonical GLB. It needs
    Blender, so it runs in UniMate's venv rather than ComfyUI's.

    Results are cached by mesh content, so re-running a graph does not redo it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh": ("STRING", {"forceInput": True}),
                "model": ("UNIMATE_MODEL",),
            },
            "optional": {
                "name": ("STRING", {"default": "", "tooltip": "Asset name; defaults "
                                                              "to the file name."}),
                "face_right": ("STRING", {"default": "", "tooltip": "Raw bone name of "
                                                                    "the RIGHT facing joint "
                                                                    "(leave empty to auto-pick)."}),
                "face_left": ("STRING", {"default": "", "tooltip": "Raw bone name of the "
                                                                   "LEFT facing joint."}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("UNIMATE_ASSET", "STRING")
    RETURN_NAMES = ("asset", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs UniMate's rig_preprocess (Blender) on a rigged mesh and caches "
                   "the result. Reports the joint labels and facing pair the model will "
                   "be conditioned on.")

    def prepare(self, rigged_mesh: str, model: dict, name: str = "",
                face_right: str = "", face_left: str = "",
                force_rebuild: bool = False):
        repo = find_repo()
        mesh = Path(rigged_mesh).expanduser()
        if not mesh.is_file():
            raise RuntimeError(f"No such mesh: {mesh}")

        asset_name = (name.strip() or mesh.stem)
        key = _digest(mesh, extra=f"{asset_name}|{face_right}|{face_left}")
        out_dir = Path(folder_paths.get_output_directory()) / "unimate_assets" / f"{asset_name}_{key}"

        if force_rebuild and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        if not (out_dir / "cond.npy").is_file():
            python = repo_python()
            cmd = [python, "-m", "data_process.rig_preprocess", "run",
                   "--input", str(mesh), "--output_dir", str(out_dir),
                   "--no_review", "--exp_dir", model["exp_dir"], "--overwrite"]
            if face_right.strip() and face_left.strip():
                cmd += ["--face_r", face_right.strip(), "--face_l", face_left.strip()]
            started = time.time()
            try:
                _run(cmd, repo, "rig_preprocess")
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Could not run {python}. rig_preprocess needs an interpreter with "
                    f"bpy installed -- normally UniMate's own venv. Set UNIMATE_PYTHON "
                    f"to it if it lives elsewhere.") from exc
            elapsed = time.time() - started
        else:
            elapsed = 0.0

        cond_name, cond = _load_cond(out_dir)
        joints = [str(n) for n in cond["joint_names"]]
        labels = [str(n) for n in cond["clean_joint_names"]]
        limit = model.get("max_joints") or 0
        if limit and len(joints) > limit:
            raise RuntimeError(
                f"{asset_name} has {len(joints)} joints but this checkpoint takes at most "
                f"{limit}. Simplify the rig (fewer finger / face / accessory bones) -- "
                f"sampling would refuse it.")

        face = cond.get("face_joint_idxs") or {}
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        vocabulary = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(counts.items()))
        report = (
            f"{cond_name}: {len(joints)} joints"
            + (f" (built in {elapsed:.0f}s)" if elapsed else " (cached)")
            + f"\nfacing pair: r={face.get('r_hip')} l={face.get('l_hip')}"
              f" body_axis={bool(face.get('body_axis'))}"
            + f"\nlabels: {vocabulary}")

        handle = {
            "dir": str(out_dir),
            "name": cond_name,
            "joints": len(joints),
            "glb": str(_canonical_glb(out_dir, cond_name)) if list(out_dir.glob("*.glb")) else "",
        }
        return (handle, report)


# ---------------------------------------------------------------------------
# 4 - Mesh Sampler
# ---------------------------------------------------------------------------

class UniMateMeshSampler:
    """Generate motion on the prepared skeleton from a text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIMATE_MODEL",),
                "asset": ("UNIMATE_ASSET",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "An object walks forward.",
                                      "tooltip": "Write like the training captions: "
                                                 "start with 'An object' and describe ONE "
                                                 "motion, not the character. "
                                                 "ONE PROMPT PER LINE chains the clip: "
                                                 "each line is another 2 s segment, "
                                                 "stitched into one long motion."}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5,
                                        "tooltip": "How strongly the prompt is followed. "
                                                   "Raise to 5-7 when a prompt is ignored; "
                                                   "lower for calmer motion."}),
                "num_repetitions": ("INT", {"default": 1, "min": 1, "max": 16,
                                            "tooltip": "Samples vary a lot -- whether a "
                                                       "creature actually travels is close "
                                                       "to a per-sample lottery. Generate "
                                                       "several and keep the best."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "chain_overlap": ("INT", {"default": 10, "min": 1, "max": 59,
                                          "tooltip": "Frames shared between consecutive "
                                                     "segments when the prompt has several "
                                                     "lines. The model blends the handover "
                                                     "across them, so raise it (15-20) when "
                                                     "a transition snaps. Ignored for a "
                                                     "single-line prompt."}),
            },
        }

    RETURN_TYPES = ("UNIMATE_MOTION", "STRING")
    RETURN_NAMES = ("motion", "info")
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs UniMate's sampler on the prepared asset. Each repetition is a "
                   "separate clip of the checkpoint's frame length (60 frames = 2 s). "
                   "Put one prompt per line to chain segments into a longer motion: "
                   "N segments give 60 + (60 - chain_overlap) * (N - 1) frames.")

    def sample(self, model: dict, asset: dict, prompt: str, cfg_scale: float,
               num_repetitions: int, seed: int, chain_overlap: int = 10):
        repo = find_repo()
        # One prompt per line. Several lines chain 60-frame segments into one
        # long motion; a single line is the plain one-clip case.
        segments = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not segments:
            raise RuntimeError("The prompt is empty.")
        text = segments[0]
        chained = len(segments) > 1

        frames = int(model.get("frames", 60)) or 60
        overlap = int(chain_overlap)
        if chained:
            if cfg_scale <= 1.0:
                raise RuntimeError(
                    "Chaining several prompts needs cfg_scale > 1.0: every segment after "
                    "the first is conditioned on its own caption, and at cfg 1.0 the "
                    "caption is zeroed. Use a single-line prompt, or raise cfg_scale.")
            if not 0 < overlap < frames:
                raise RuntimeError(
                    f"chain_overlap must be between 1 and {frames - 1} for this "
                    f"checkpoint ({frames}-frame window); got {overlap}.")

        stamp = f"{int(time.time() * 1000):x}"
        out_dir = Path(folder_paths.get_output_directory()) / "unimate_motion" / stamp
        # Prefer ComfyUI's interpreter so the GPU work happens in the process
        # tree ComfyUI manages. Its only missing piece is `tyro`, the CLI's
        # argument parser (requirements.txt installs it); without it, fall back
        # to UniMate's own venv, which has the full set.
        python = sys.executable
        if not _can_import(python, "tyro"):
            python = repo_python()
            if not _can_import(python, "tyro"):
                raise RuntimeError(
                    "Neither ComfyUI's interpreter nor UniMate's venv can import `tyro`, "
                    "which UniMate's sampler CLI needs. Install this pack's "
                    "requirements.txt (`pip install tyro`) into ComfyUI's venv.")
        cmd = [python, "-m", "unimate.inference.sample",
               "--exp_dir", model["exp_dir"],
               "--model_path", model["checkpoint"],
               "--asset", asset["dir"],
               # `--prompt` is a variadic tuple: every segment follows the one
               # flag. Repeating the flag keeps only the last value, which
               # silently collapses a chain back to a single clip.
               "--prompt", *segments,
               "--num_repetitions", str(int(num_repetitions)),
               "--cfg_scale", str(float(cfg_scale)),
               "--seed", str(int(seed)),
               "--only_save_motion",
               "--output_dir", str(out_dir)]
        if chained:
            cmd += ["--motion_expand", "--expand_overlap", str(overlap)]
        # An explicit env rather than mutating os.environ: two prompts sampling
        # at once would otherwise clobber each other's PYTHONPATH. UniMate logs
        # a '->' that cp1252 consoles cannot encode, hence the forced UTF-8.
        prev = os.environ.get("PYTHONPATH", "")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(repo), str(repo / "Motion"), prev) if p)
        env["PYTHONIOENCODING"] = "utf-8"
        started = time.time()
        _run(cmd, repo, "unimate.inference.sample", env=env)

        # A chained run lands in a motion_expand/ sibling of the plain output.
        motions = out_dir / ("motion_expand/motions" if chained else "motions")
        clips = sorted(motions.glob("*.npy"))
        if not clips:
            raise RuntimeError(
                f"The sampler wrote no motions into {motions}. If the asset has more "
                f"joints than the checkpoint takes it refuses silently -- check the "
                f"preparation report.")
        handle = {
            "clips": [str(p) for p in clips],
            "asset_dir": asset["dir"],
            "fps": 30.0,
            "prompt": text,
            "segments": segments,
        }
        lines = []
        for clip in clips:
            n = int(np.load(clip, allow_pickle=True).shape[0])
            lines.append(f"{clip.name}: {n} frames, {n / 30.0:.2f}s")
        head = (f"{len(clips)} clip(s) in {time.time() - started:.0f}s | cfg {cfg_scale} "
                f"| seed {seed}")
        if chained:
            head += f"\n{len(segments)} segments chained, overlap {overlap}"
        info = head + "\n" + "\n".join(lines)
        return (handle, info)


# ---------------------------------------------------------------------------
# 5 - Mesh Export
# ---------------------------------------------------------------------------

class UniMateMeshExport:
    """Write the generated motion onto the asset's mesh as an animated GLB.

    Does not use Blender. UniMate's own mesh driving does, and is currently
    broken on Blender 4.4+ (``action.fcurves`` was removed by the slotted-action
    API); this writes the glTF animation directly, which also keeps ComfyUI's
    venv free of bpy. See glb_anim.py for how the two rigs' bone conventions
    are reconciled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset": ("UNIMATE_ASSET",),
                "motion": ("UNIMATE_MOTION",),
                "filename_prefix": ("STRING", {"default": "3D/UniMate_"}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "STRING", "STRING")
    RETURN_NAMES = ("model_3d", "glb_path", "info")
    OUTPUT_TOOLTIPS = (
        "The saved animated GLB as a 3D file: connect to Preview 3D. One per clip.",
        "GLB path relative to the output folder. One per clip.",
        "What was written: frames, duration and track counts.",
    )
    # model_3d and glb_path are one per generated clip; info describes them all,
    # so a downstream Preview 3D runs once per clip while the text stays single.
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Writes each generated clip onto the asset's canonical mesh as a GLB."

    def export(self, asset: dict, motion: dict, filename_prefix: str, fps: float = 30.0):
        ensure_importable()
        from Animation import Animation, positions_global, rotations_global, transforms_global
        from Quaternions import Quaternions
        from unimate.utils.motion_utils import recover_unimate_anim_from_rot

        asset_dir = Path(asset["dir"])
        cond_name, cond = _load_cond(asset_dir)
        glb = asset.get("glb") or str(_canonical_glb(asset_dir, cond_name))
        parents = np.asarray(cond["parents"])
        # FK is built on tpos_offsets, NOT offsets: the first are bone vectors in
        # the canonical frame, the second parent-relative translations, and
        # swapping them yields a pose rotated about 90 degrees with no error.
        tpos_offsets = np.asarray(cond["tpos_offsets"])

        rest_anim = Animation(
            rotations=Quaternions.id((1, len(parents))),
            positions=tpos_offsets[None],
            orients=Quaternions.id(len(parents)),
            offsets=tpos_offsets,
            parents=parents,
        )
        rest_world = transforms_global(rest_anim)[0]

        out_root = Path(folder_paths.get_output_directory())
        # Resolved once: get_save_image_path advances the counter itself, so
        # calling it per clip and then adding the index skips numbers.
        full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(out_root))
        Path(full).mkdir(parents=True, exist_ok=True)

        models, relative, previews, lines = [], [], [], []
        for index, clip_path in enumerate(motion["clips"]):
            data = np.load(clip_path)
            anim = recover_unimate_anim_from_rot(data, parents, tpos_offsets)
            positions = positions_global(anim)
            rotations = rotations_global(anim).transforms()
            world = np.zeros(positions.shape[:2] + (4, 4))
            world[..., :3, :3] = rotations
            world[..., :3, 3] = positions
            world[..., 3, 3] = 1.0

            name = f"{filename}_{counter + index:05}_"
            out_path = Path(full) / f"{name}.glb"
            report = motion_to_glb(glb, cond, world, rest_world, out_path,
                                   fps=float(fps),
                                   clip_name=Path(clip_path).stem[:60])

            models.append(File3D(str(out_path), file_format="glb")
                          if File3D is not None else str(out_path))
            relative.append(os.path.join(subfolder, f"{name}.glb").replace("\\", "/"))
            previews.append({"filename": f"{name}.glb", "subfolder": subfolder,
                             "type": "output"})
            lines.append(f"{name}.glb: {report['frames']} frames, "
                         f"{report['duration']:.2f}s, {report['rotation_tracks']} rotation "
                         f"+ {report['translation_tracks']} translation tracks")

        info = f"{len(models)} GLB(s)\n" + "\n".join(lines)
        # OUTPUT_NODE: the "3d" ui payload is what makes the node show the
        # result itself, the same contract Save/Preview 3D nodes use.
        return {"ui": {"3d": previews}, "result": (models, relative, info)}


NODE_CLASS_MAPPINGS = {
    "UniMateModelLoader": UniMateModelLoader,
    "UniMateMeshLoader": UniMateMeshLoader,
    "UniMateMeshPreparation": UniMateMeshPreparation,
    "UniMateMeshSampler": UniMateMeshSampler,
    "UniMateMeshExport": UniMateMeshExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniMateModelLoader": "UniMate Model Loader",
    "UniMateMeshLoader": "UniMate Mesh Loader",
    "UniMateMeshPreparation": "UniMate Mesh Preparation",
    "UniMateMeshSampler": "UniMate Mesh Sampler",
    "UniMateMeshExport": "UniMate Mesh Export",
}
