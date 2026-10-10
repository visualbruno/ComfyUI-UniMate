# ComfyUI-UniMate

Text-driven motion for **any** rigged skeleton — quadrupeds, birds, insects,
dragons, humanoids — using [UniMate](https://github.com/Friedrich-M/UniMate)
(SIGGRAPH Asia 2026). Give it a rigged mesh and a prompt; get an animated GLB.

```
UniMate Model Loader ─┐
UniMate Mesh Loader ──> Mesh Preparation ──> Mesh Sampler ──> Mesh Export ──> .glb
```

## ⚠️ Licence — read before using the output

The **code here is MIT**. The **UniMate checkpoints are CC BY-NC 4.0**, and
their [model card](https://huggingface.co/Linzhan/UniMate) states:

> We make no warranty that the checkpoints, **or motions generated with them**,
> are free of third-party rights.

and flags that the provenance of the Truebones-trained (animal) portion is
disputed. The authors' own position, asked directly: they *"wouldn't recommend
shipping them in a commercial product."*

So: fine for evaluation, prototyping and non-commercial work. **Not** for
commercial assets. These nodes are tooling; they do not change the terms of the
weights you point them at.

## Install

1. Clone this into `ComfyUI/custom_nodes/`.
2. Clone UniMate somewhere and tell the pack where it is — set `UNIMATE_REPO`,
   or write the path into `unimate_path.txt` beside this README. `C:/Git/UniMate`
   and `~/UniMate` are found automatically.
3. `pip install -r requirements.txt` into ComfyUI's venv (just `tyro`, the
   sampler CLI's argument parser). Without it the pack falls back to UniMate's
   own venv, which also works.
4. Download a checkpoint run from
   [Linzhan/UniMate](https://huggingface.co/Linzhan/UniMate) into
   `ComfyUI/models/unimate/<run>/`, keeping its layout:

   ```
   models/unimate/unimate_uniml3d_f60_v3/
       config.json
       dataset_stats.npy
       checkpoints/checkpoint_step_150000.pt
   ```

**Preparation needs Blender.** It runs `rig_preprocess` in UniMate's own venv
(where `bpy` lives) rather than installing a ~1 GB wheel into ComfyUI's. If
UniMate's venv is elsewhere, point `UNIMATE_PYTHON` at an interpreter that can
`import bpy`.

## The nodes

| Node | Does | Cost |
|---|---|---|
| **Model Loader** | Picks a checkpoint run; reports its joint/frame limits | instant |
| **Mesh Loader** | Picks a rigged GLB/GLTF/FBX. Takes ComfyUI-SkinToken's `rigged_path` directly | instant |
| **Mesh Preparation** | `rig_preprocess`: labels joints, picks the facing pair, bakes a canonical rest pose. **Cached by mesh content** | ~30 s, once per mesh |
| **Mesh Sampler** | Generates motion from a prompt. One prompt per line chains longer motions | ~20–30 s per segment |
| **Mesh Export** | Writes each clip onto the mesh as an animated GLB. Output node — saves to `output/`, shows the result in the node and emits `FILE_3D_GLB` for **Preview 3D** | instant |

## Things that will bite you

**Prompt like the training captions.** Start with *"An object"* and describe
one motion, not the character: `"An object walks forward."`, not `"A red dragon
flies over a castle."` All 13,769 training captions begin that way.

**There is a hard joint limit** — 71 on the v3 checkpoints. A rig with more is
refused outright, and `rig_preprocess` will not prune for you. Preparation
reports the count and fails early rather than letting the sampler refuse
silently.

**Samples vary a lot.** Whether a creature actually travels is close to a
per-sample lottery — measured travel ranged 0.02 to 3.4 across four samples of
one prompt. Use `num_repetitions` and keep the best.

**`cfg_scale` is the strongest knob.** 3 is the default; 5–7 makes a prompt
that is being ignored take hold. On a spider, raising 3→6 doubled the per-frame
foot motion.

**Clips are 2 seconds** (60 frames at 30 fps). That is the model's window, and it
is fixed: every checkpoint was trained at `max_motion_length = 60`.

**To go longer, put one prompt per line.** Each line is another 2 s segment, and
the sampler chains them into a single motion (UniMate's `--motion_expand`):

```
An object walks forward.
An object turns to the left.
An object jumps.
```

Segment *n* has its first `chain_overlap` frames pinned to the last
`chain_overlap` frames of segment *n-1*, and the flow ODE denoises the rest, so
continuity comes from the sampler rather than from blending afterwards. Length is
`60 + (60 - chain_overlap) * (N - 1)` frames — at the default overlap of 10:
2 s, 3.7 s, 5.3 s, 7 s for 1, 2, 3, 4 lines.

Three things to know about it. It is a **storyboard, not a stretch**: each line
is a fresh generation, so repeat a line to extend one action. It is **strictly
sequential** (segment *n* needs *n-1*), so it does not batch — budget one
sampling pass per line. And it needs **`cfg_scale` > 1.0**, because at 1.0 the
caption is zeroed and the per-segment prompts would do nothing; the node refuses
rather than silently generating noise. Raise `chain_overlap` to 15–20 if a
transition snaps.

**Check the preparation report.** The joint labels and facing pair are what the
model is conditioned on. The offline labeller is decent but not perfect; a
wrong facing pair rotates the whole creature. You can override it with
`face_right` / `face_left` (raw bone names).

## Notes on the implementation

**Export does not use Blender.** UniMate's own mesh driving does, and it is
broken on Blender 4.4+: `blender_rig.py` calls `action.fcurves.new`, which the
slotted-action API removed. `glb_anim.py` writes the glTF animation directly.

The subtle part is that `rig_preprocess` bakes a canonical GLB whose rest pose
matches the cond's T-pose exactly (measured at 3e-07) but whose per-bone axes
are Blender's, so each joint's rest rotation differs by about 90°. Rather than
reproduce UniMate's axis-conjugation, the exporter drives **world** transforms:

```
world_glb(f) = world_motion(f) @ inv(world_motion(rest)) @ world_glb(rest)
```

applying the motion's delta from its own rest pose, so the convention cancels.
Verified at 3.6e-07 – 7.4e-07 against the forward kinematics it is given, on a
53-joint horse, a 41-joint spider and a 71-joint dragon.

**One trap worth repeating:** a cond carries both `offsets` (parent-relative
translations) and `tpos_offsets` (bone vectors in the canonical frame). UniMate's
FK wants `tpos_offsets`. Passing `offsets` returns a pose rotated ~90° and
raises nothing — it decodes a level quadruped as a vertical one.

## Credits

UniMate by Linzhan Mou, Jiahui Lei, Zhiyang Dou, Chenyue Cai, Chaoyue Song,
Adam Finkelstein and Szymon Rusinkiewicz (Princeton · UC Berkeley · MIT · NTU).
