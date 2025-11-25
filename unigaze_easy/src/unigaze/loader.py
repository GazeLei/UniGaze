# unigaze/loader.py
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load



# ---- import your model(s)
from .models.mae_gaze import MAE_Gaze
from .models.mae import interpolate_pos_embed  

MODEL_INDEX = {
    "unigaze_b16_joint": {
        "repo_id": "UniGaze/UniGaze-models",
        "filename": "unigaze_b16_joint.safetensors",
        "revision": "main",
        "builder": "unigaze_b16_joint",
        "model_kwargs": {"model_type": "vit_b_16"},
    },

    "unigaze_l16_joint": {
        "repo_id": "UniGaze/UniGaze-models",
        "filename": "unigaze_l16_joint.safetensors",
        "revision": "main",
        "builder": "unigaze_l16_joint",
        "model_kwargs": {"model_type": "vit_l_16"},
    },
    

    "unigaze_h14_joint": {
        "repo_id": "UniGaze/UniGaze-models",
        "filename": "unigaze_h14_joint.safetensors",
        "revision": "main",
        "builder": "unigaze_h14_joint",
        "model_kwargs": {"model_type": "vit_h_14"},
    },

    "unigaze_h14_cross_X": {
        "repo_id": "UniGaze/UniGaze-models",
        "filename": "unigaze_h14_cross_X.safetensors",
        "revision": "main",
        "builder": "unigaze_h14_cross_X",
        "model_kwargs": {"model_type": "vit_h_14"},
    },
    
}


def build_unigaze_model(builder_key: str, **overrides):
    """
    Create a model instance by key. You can extend this to more families later.
    `overrides` lets users pass constructor kwargs (e.g., global_pool=True).
    """
    if builder_key not in MODEL_INDEX:
        raise KeyError(f"Unknown builder '{builder_key}'")
    
    base = MODEL_INDEX[builder_key].get("model_kwargs", {})
    return MAE_Gaze(**{**base, **overrides, "custom_pretrained_path": None})
  
    


def load(name: str, device: str = "cpu", **kwargs):
    """
    Create and (optionally) load pretrained weights for a named model.
    Extra kwargs override constructor defaults (e.g., global_pool=True).
    """
    spec = MODEL_INDEX[name]
    model = build_unigaze_model(spec["builder"], **kwargs)
    if name.startswith("unigaze"):
        path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            revision=spec["revision"],
            local_files_only=bool(int(os.getenv("HF_HUB_OFFLINE", "0"))),
        )
        model.load_unigaze_weights(path)
    elif name.startswith("mae"):
        ## TODO: this is not supported yet
        path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            revision=spec["revision"],
            local_files_only=bool(int(os.getenv("HF_HUB_OFFLINE", "0"))),
        )
        model.load_pretrained_mae_weights(path)

    return model.to(device).eval()
