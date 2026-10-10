"""ComfyUI-UniMate: text-driven motion for any rigged skeleton.

Registers ``models/unimate`` so checkpoint runs sit where every other model
pack puts theirs, then exposes the five nodes.
"""

import os

import folder_paths

UNIMATE_MODEL_DIR = os.path.join(folder_paths.models_dir, "unimate")

try:
    os.makedirs(UNIMATE_MODEL_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("unimate", UNIMATE_MODEL_DIR, is_default=True)
except Exception:
    # A read-only models directory is not a reason to fail the import; the
    # nodes still work with an absolute run_path.
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
