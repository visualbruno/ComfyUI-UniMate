"""Write a generated motion into an asset's canonical GLB, without Blender.

UniMate's own mesh driving goes through ``bpy``, which (a) is a heavy
dependency to put in ComfyUI's venv and (b) is currently broken on Blender
4.4+: ``data_process/utils/blender_rig.py`` calls ``action.fcurves.new``, which
the slotted-action API removed, so it raises ``'Action' object has no attribute
'fcurves'`` on any modern Blender. This writes the glTF animation directly
instead.

**Why world transforms rather than the motion's own local rotations.**
``rig_preprocess`` bakes a canonical GLB whose rest pose matches the cond's
T-pose exactly -- measured at 3e-07 across every asset tested -- but whose
*per-bone axes* are Blender's (bones point down +Y), so each joint's rest
rotation differs from the motion's by about 90 degrees. Writing the motion's
local rotations straight onto those nodes would therefore twist every bone,
and correcting it means reproducing UniMate's ``motion_axis_alignment``
conjugation. Driving world transforms sidesteps the question: for each joint,

    world_glb(f) = world_motion(f) @ inv(world_motion(rest)) @ world_glb(rest)

applies the motion's *delta from its own rest pose* to the GLB's rest pose, so
whatever axis convention each rig uses cancels. The node-local transform that
produces it is then ``inv(world_glb(parent, f)) @ world_glb(f)``.

Skinning stays correct because the GLB's ``inverseBindMatrices`` are relative
to its own rest pose, which is exactly the pose this formula leaves untouched
at frame zero.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Local translations that never move do not need a track; this is how far a
# joint may drift (in canonical units, on a skeleton scaled to diameter 2)
# before we decide it genuinely translates and write one.
TRANSLATION_EPS = 1e-6


# ---------------------------------------------------------------------------
# glTF container
# ---------------------------------------------------------------------------

def read_glb(path) -> Tuple[dict, bytes]:
    """Return ``(json, binary chunk)`` of a .glb."""
    blob = Path(path).read_bytes()
    if blob[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF (.glb)")
    total = struct.unpack("<I", blob[8:12])[0]
    offset, gltf, binary = 12, None, b""
    while offset < min(total, len(blob)):
        length, kind = struct.unpack("<II", blob[offset:offset + 8])
        body = blob[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:          # 'JSON'
            gltf = json.loads(body)
        elif kind == 0x004E4942:        # 'BIN\0'
            binary = body
        offset += 8 + length + (-length % 4)
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")
    return gltf, binary


def write_glb(path, gltf: dict, binary: bytes) -> None:
    """Write ``(json, binary)`` back out, with the padding the spec requires."""
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)             # JSON pads with spaces
    blob = binary + b"\0" * (-len(binary) % 4)  # BIN pads with zeros
    total = 12 + 8 + len(text) + (8 + len(blob) if blob else 0)
    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    if blob:
        out += struct.pack("<II", len(blob), 0x004E4942) + blob
    Path(path).write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> glTF (x, y, z, w), via the stable branch."""
    t = float(np.trace(m))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) or 1.0)


def node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = quat_xyzw_to_matrix(node.get("rotation", [0, 0, 0, 1]))
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag(np.asarray(node["scale"], dtype=np.float64))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def compose(rot3: np.ndarray, pos3: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot3
    m[:3, 3] = pos3
    return m


def node_parents(gltf: dict) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, node in enumerate(gltf.get("nodes", [])):
        for child in node.get("children", []):
            out[child] = i
    return out


def world_matrices(gltf: dict, parents: Dict[int, int]) -> Dict[int, np.ndarray]:
    """Rest world matrix of every node, ancestors composed to the scene root."""
    nodes = gltf.get("nodes", [])
    cache: Dict[int, np.ndarray] = {}

    def walk(i: int) -> np.ndarray:
        if i not in cache:
            local = node_local_matrix(nodes[i])
            p = parents.get(i)
            cache[i] = local if p is None else walk(p) @ local
        return cache[i]

    for i in range(len(nodes)):
        walk(i)
    return cache


# ---------------------------------------------------------------------------
# Buffer building
# ---------------------------------------------------------------------------

class _BufferBuilder:
    """Appends float arrays to the binary chunk, returning accessor indices."""

    def __init__(self, gltf: dict, binary: bytes):
        self.gltf = gltf
        self.chunks: List[bytes] = [binary]
        self.offset = len(binary)
        gltf.setdefault("bufferViews", [])
        gltf.setdefault("accessors", [])
        gltf.setdefault("buffers", [{"byteLength": 0}])

    def add(self, array: np.ndarray, kind: str) -> int:
        """Append ``array`` (float32) as a new accessor of glTF ``kind``."""
        data = np.ascontiguousarray(array, dtype=np.float32)
        pad = -self.offset % 4
        if pad:
            self.chunks.append(b"\0" * pad)
            self.offset += pad
        raw = data.tobytes()
        self.gltf["bufferViews"].append(
            {"buffer": 0, "byteOffset": self.offset, "byteLength": len(raw)})
        self.chunks.append(raw)
        self.offset += len(raw)

        count = data.shape[0]
        accessor = {
            "bufferView": len(self.gltf["bufferViews"]) - 1,
            "componentType": 5126,          # FLOAT
            "count": int(count),
            "type": kind,
        }
        if kind == "SCALAR":
            # The spec requires min/max on an animation sampler's input.
            accessor["min"] = [float(data.min())]
            accessor["max"] = [float(data.max())]
        self.gltf["accessors"].append(accessor)
        return len(self.gltf["accessors"]) - 1

    def finish(self) -> bytes:
        binary = b"".join(self.chunks)
        self.gltf["buffers"][0]["byteLength"] = len(binary)
        self.gltf["buffers"][0].pop("uri", None)
        return binary


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def motion_to_glb(canonical_glb, cond: dict, motion_world: np.ndarray,
                  motion_rest_world: np.ndarray, out_path, fps: float = 30.0,
                  clip_name: str = "UniMate") -> dict:
    """Write one generated motion into *canonical_glb* as a glTF animation.

    Args:
        canonical_glb: the asset's ``<name>.glb`` from ``rig_preprocess``.
        cond: that asset's cond entry (joint names, parents, rest pose).
        motion_world: ``(T, J, 4, 4)`` world matrices per frame per joint, in
            the cond's joint order. The caller builds these with UniMate's own
            forward kinematics rather than a reimplementation here -- the
            offsets a recovered ``Animation`` is built on are ``tpos_offsets``,
            not ``offsets``, and mixing the two silently yields a pose rotated
            about 90 degrees.
        out_path: .glb to write.
        fps: playback rate the key times are built from.

    Returns a small report: frame count, duration, and how many joints were
    matched, translated and rotated.
    """
    gltf, binary = read_glb(canonical_glb)
    skins = gltf.get("skins") or []
    if not skins:
        raise ValueError(f"{canonical_glb} has no skin: it is not a rigged mesh")

    nodes = gltf["nodes"]
    joints = skins[0]["joints"]
    glb_names = [nodes[j].get("name", f"node_{j}") for j in joints]
    cond_names = [str(n) for n in cond["joint_names"]]

    # The cond's joint order and the skin's differ (BFS vs however the
    # exporter wrote them), so everything below is matched by NAME.
    index_of = {name: node for name, node in zip(glb_names, joints)}
    missing = [n for n in cond_names if n not in index_of]
    if missing:
        raise ValueError(
            f"{Path(canonical_glb).name} is missing {len(missing)} of the motion's "
            f"joints (e.g. {missing[:3]}): it is not the asset this motion was made for")

    parents_map = node_parents(gltf)
    rest_world = world_matrices(gltf, parents_map)
    cond_parents = np.asarray(cond["parents"])
    motion_world = np.asarray(motion_world, dtype=np.float64)
    n_frames, n_joints = motion_world.shape[0], motion_world.shape[1]
    if n_joints != len(cond_names):
        raise ValueError(
            f"motion has {n_joints} joints but the cond names {len(cond_names)}")

    # --- the motion's rest pose ---------------------------------------------
    # Supplied by the caller from the SAME forward kinematics that produced
    # `motion_world`. The cond's `tpos_global_rotations` cannot stand in for
    # it: those are in the export's bone convention, not the Animation
    # library's, and using them puts a fixed rotation into every joint.
    motion_rest = np.asarray(motion_rest_world, dtype=np.float64)
    if motion_rest.shape[0] != n_joints:
        raise ValueError(
            f"rest pose has {motion_rest.shape[0]} joints, motion has {n_joints}")

    # --- per joint: the constant that maps motion space onto the GLB's -------
    # A_j = inv(world_motion_rest_j) @ world_glb_rest_j
    bridge = {}
    for j, name in enumerate(cond_names):
        bridge[j] = np.linalg.inv(motion_rest[j]) @ rest_world[index_of[name]]

    # --- node-local TRS per frame -------------------------------------------
    times = (np.arange(n_frames, dtype=np.float32) / float(fps))
    rotations: Dict[int, np.ndarray] = {}
    translations: Dict[int, np.ndarray] = {}
    for j, name in enumerate(cond_names):
        node = index_of[name]
        parent_node = parents_map.get(node)
        # A joint's glTF parent may be a plain node (an armature wrapper); its
        # world matrix is constant unless it is itself an animated joint.
        parent_cond = int(cond_parents[j])
        rot = np.empty((n_frames, 4), dtype=np.float32)
        pos = np.empty((n_frames, 3), dtype=np.float32)
        for f in range(n_frames):
            world = motion_world[f, j] @ bridge[j]
            if parent_cond >= 0:
                parent_world = motion_world[f, parent_cond] @ bridge[parent_cond]
            elif parent_node is not None:
                parent_world = rest_world[parent_node]
            else:
                parent_world = np.eye(4)
            local = np.linalg.inv(parent_world) @ world
            basis = local[:3, :3]
            scale = np.linalg.norm(basis, axis=0)
            scale[scale == 0] = 1.0
            rot[f] = matrix_to_quat_xyzw(basis / scale)
            pos[f] = local[:3, 3]
        rotations[node] = rot
        # Only the root really translates; a child's local translation is its
        # constant bone offset. Writing a flat track for 70 joints would bloat
        # the file for nothing, so one is written only if it actually moves.
        if float(np.ptp(pos, axis=0).max()) > TRANSLATION_EPS:
            translations[node] = pos

    # --- assemble the animation ---------------------------------------------
    builder = _BufferBuilder(gltf, binary)
    time_accessor = builder.add(times, "SCALAR")
    samplers, channels = [], []
    for node, rot in rotations.items():
        samplers.append({"input": time_accessor, "interpolation": "LINEAR",
                         "output": builder.add(rot, "VEC4")})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": node, "path": "rotation"}})
    for node, pos in translations.items():
        samplers.append({"input": time_accessor, "interpolation": "LINEAR",
                         "output": builder.add(pos, "VEC3")})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": node, "path": "translation"}})

    gltf.setdefault("animations", []).append(
        {"name": clip_name, "samplers": samplers, "channels": channels})
    write_glb(out_path, gltf, builder.finish())

    return {
        "frames": n_frames,
        "duration": float(times[-1]) if n_frames else 0.0,
        "joints": n_joints,
        "rotation_tracks": len(rotations),
        "translation_tracks": len(translations),
        "path": str(out_path),
    }
