import torch
import os
import math
import folder_paths

# -------------------------------------------------------------------
# Optional safetensors support flag (keep both names for legacy checks)
# -------------------------------------------------------------------
try:
    from safetensors.torch import load_file as safe_load, save_file as safe_save
    SAFETENSORS = True
    safetensors_available = True   # legacy alias
except ImportError:
    SAFETENSORS = False
    safetensors_available = False
    safe_load = None
    safe_save = None

# -------------------------------------------------------------------
# Build default output directory: <ComfyUI>/models/loras/merged-loras
# -------------------------------------------------------------------
lora_base_path = folder_paths.get_folder_paths("loras")[0]
OUTPUT_DIR = os.path.join(lora_base_path, "merged-loras")
# print(OUTPUT_DIR)  # ['/path/to/ComfyUI/models/loras/merged-loras'] 

# =============================================================================
# OnlyLoadLoRAsModel
# =============================================================================
class OnlyLoadLoRAsModel:
    """Load a single LoRA file from <ComfyUI>/models/loras.

    * `category_filter` – folder drop‑down (handled by front‑end JS)
    * `lora_name`       – file selector (full list, filtered on the client)
    """

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        dirs  = sorted({os.path.dirname(p) for p in names if os.path.dirname(p)})
        return {
            "required": {
                "category_filter": (["All"] + dirs,),
                "lora_name":       (names,),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LoRA"

    def load(self, lora_name, category_filter='All'):
        # Front‑end JS handles the folder filtering; the back‑end only loads the file.
        lora_path = folder_paths.get_full_path('loras', lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(lora_name)

        if lora_path.endswith('.safetensors'):
            if not SAFETENSORS or safe_load is None:
                raise ImportError('pip install safetensors to load .safetensors')
            state_dict = safe_load(lora_path, device='cpu')
        else:
            state_dict = torch.load(lora_path, map_location='cpu')
        return (state_dict,)

# =============================================================================
# MergeLoRAsKohyaSSLike
# =============================================================================
class MergeLoRAsKohyaSSLike:
    """Merge multiple LoRA state‑dicts (kohya‑ss / SuperMerger style).

    * **Order‑independent** – A+B == B+A when ratios are the same.
    * `force_same_strength=yes`  ⇒   ratio → √ratio (matches Web‑UI "Strength").
    * Per‑module scaling by √(αᵢ / avgα) for more balanced feature retention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "weight1": ("FLOAT", {"default": 1.00,"step": 0.01}),
                "model2": ("MODEL",),
                "weight2": ("FLOAT", {"default": 1.00,"step": 0.01}),
                "weight3": ("FLOAT", {"default": 0.00,"step": 0.01}),
                "weight4": ("FLOAT", {"default": 0.00,"step": 0.01}),
                "force_same_strength": (["no", "yes"], {"default": "no"}),
                "allow_overwrite": (["no", "yes"], {"default": "no"}),
                "save_dtype": (["fp16", "float", "bf16"], {"default": "fp16"}),
                "reset_dim": (["no", "auto", "4", "8", "16", "32", "64"], {"default": "no"})
            },
            "optional": {
                "model3": ("MODEL",),
                "model4": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge"
    CATEGORY = "LoRA"

    # ---------- helpers ----------
    def _safe_scalar(self, t: torch.Tensor):
        """Return float value regardless of dtype/shape."""
        if t.numel() == 1:
            return t.float().item()
        return t.float().mean().item()
    
    # ---------- main ----------
    def merge(self, model1, weight1, model2, weight2,
              weight3, weight4, force_same_strength, allow_overwrite,
              save_dtype, reset_dim, model3=None, model4=None):

        models_with_w = [(model1, weight1), (model2, weight2)]
        if model3 is not None:
            models_with_w.append((model3, weight3))
        if model4 is not None:
            models_with_w.append((model4, weight4))

        # remove zero-weight items
        models_with_w = [(sd, w) for sd, w in models_with_w if w != 0]

        # ---- gather α for each module ----
        module_alphas_list = []
        merged_base_alpha = {}

        for sd, _ in models_with_w:
            alphas = {}
            # read explicit .alpha keys
            for k, v in sd.items():
                if k.endswith('.alpha'):
                    module = k[:-6]
                    alphas[module] = self._safe_scalar(v)
            # fallback: use dim from lora_down weight
            for k, v in sd.items():
                if '.lora_down' in k:
                    module = k.split('.lora_down')[0]
                    if module not in alphas:
                        alphas[module] = v.size(0)
            module_alphas_list.append(alphas)
            # accumulate
            for m, a in alphas.items():
                s, c = merged_base_alpha.get(m, (0.0, 0))
                merged_base_alpha[m] = (s + a, c + 1)

        # average to get base α
        for m, (s, c) in merged_base_alpha.items():
            merged_base_alpha[m] = s / c

        # dtype
        dtype_map = {"fp16": torch.float16, "float": torch.float32, "bf16": torch.bfloat16}
        final_dtype = dtype_map[save_dtype]

        # --- 2. Actual merging ---
        merged_sd: dict[str, torch.Tensor] = {}
        for (sd, ratio), mod_alpha in zip(models_with_w, module_alphas_list):
            if force_same_strength == "yes":
                ratio = math.copysign(math.sqrt(abs(ratio)), ratio)
            for k, tensor in sd.items():
                if k.endswith('.alpha') or '.lora_' not in k:
                    continue
                module = k.split('.lora_')[0]
                alpha_i = mod_alpha.get(module, merged_base_alpha[module])
                base_alpha = merged_base_alpha[module]
                scale = math.sqrt(alpha_i / base_alpha) * ratio
                # keep positive for up weight when ratio negative (matches supermerger)
                if 'lora_up' in k and scale < 0:
                    scale = abs(scale)
                contrib = tensor.float() * scale
                merged_sd[k] = contrib if k not in merged_sd else merged_sd[k] + contrib

        # --- 3. Write back averaged α keys ---
        for module, base_alpha in merged_base_alpha.items():
            merged_sd[f"{module}.alpha"] = torch.tensor(base_alpha, dtype=final_dtype)

        # cast
        for k in list(merged_sd.keys()):
            if merged_sd[k].dtype != final_dtype:
                merged_sd[k] = merged_sd[k].to(dtype=final_dtype)

        return (merged_sd, )

# =============================================================================
# SaveLoRAModels
# =============================================================================
class SaveLoRAModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_model": ("MODEL", ),
                "modeloutput": ("STRING", {"default": "merged_lora.safetensors"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "LoRA"

    def save(self, merged_model, modeloutput):
        if not os.path.isabs(modeloutput):
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            modeloutput = os.path.join(OUTPUT_DIR, modeloutput)
        if modeloutput.endswith('.safetensors') and safetensors_available and safe_save is not None:
            safe_save(merged_model, modeloutput)
        else:
            torch.save(merged_model, modeloutput)
        print(f"LoRA model saved to {modeloutput}")
        return (modeloutput,)
    
# =============================================================================
# Node registration
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "OnlyLoadLoRAsModel": OnlyLoadLoRAsModel,
    "MergeLoRAsKohyaSSLike": MergeLoRAsKohyaSSLike,
    "SaveLoRAModels": SaveLoRAModels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnlyLoadLoRAsModel": "Load LoRA Model(merge)",
    "MergeLoRAsKohyaSSLike": "Merge LoRA Models(merge)",
    "SaveLoRAModels": "Save LoRA Model(merge)",
}
