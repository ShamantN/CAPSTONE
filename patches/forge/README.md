# MMCCP Upstream Forge Patches (`dfdcbab685e57677014f05a3309b48cc87383167`)

These **3 patches** capture the exact code modifications applied on the reference Windows workstation (`C:\Stable-Diffusion\webui`) against upstream `lllyasviel/stable-diffusion-webui-forge` commit **`dfdcbab685e57677014f05a3309b48cc87383167`** (`2025-06-26`).

- **Tooling:** Apply, check, or revert cleanly with [`tools/apply_forge_patches.py`](../../tools/apply_forge_patches.py):
  ```powershell
  python tools\apply_forge_patches.py --forge-dir "C:\Stable-Diffusion" --check
  python tools\apply_forge_patches.py --forge-dir "C:\Stable-Diffusion" --apply
  python tools\apply_forge_patches.py --forge-dir "C:\Stable-Diffusion" --revert
  ```
- **Safe target:** The script verifies `.git` HEAD == `dfdcbab685e57677014f05a3309b48cc87383167` before running `git apply --check` and refuses if your Forge commit differs.

---

## Patch Inventory & Why Each Exists

### 1. `01_ipadapter_device_and_np_fix.patch`
- **Target file in Forge:** `extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py`
- **Upstream defects fixed:**
  1. Missing `import numpy as np` inside `IPAdapterPlus.py`, which caused `NameError: name 'np' is not defined` inside `tensorToNP()` (`experiments/phase9_master_completion_report.md` lines 12 & 75).
  2. Dict/ndarray/`None` handling in `tensorToNP()` so WebUI `/sdapi/v1/*` dictionary image payloads (`{'image': np.ndarray}`) unwrap instead of crashing (`experiments/phase8_4_master_validation_report.md` line 58).
  3. Cross-attention tensor device/dtype synchronization in `CrossAttentionPatch.__call__()` (`target_device = q.device`, `target_dtype = org_dtype` for `cur_cond`, `cur_uncond`, `k_layer`, `v_layer`, `attention_function`, and interpolated `mask`), preventing `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!` when 8 GB GPUs offload layers between RAM and VRAM (`experiments/ipadapter_device_fix_report.md` lines 15–60).

### 2. `02_controlnet_null_param_guard.patch`
- **Target file in Forge:** `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py`
- **Why it exists:** Adds `if i in self.current_params:` membership guards inside `ControlNetForForgeOfficial.process_before_every_sampling()` and `postprocess_batch_list()`.
- **What breaks without it:** When multiple ControlNet units are passed via API payloads or partially skipped/unpopulated between Pass 1 (`txt2img` with OpenPose + IP-Adapter) and Pass 2 (`img2img`), indexing `self.current_params[i]` raises `KeyError: i` if unit `i` was omitted from `current_params`.

### 3. `03_forge_utils_numpy_to_pytorch.patch`
- **Target file in Forge:** `modules_forge/utils.py`
- **Why it exists:** Unwraps `{'image': ...}` dictionaries, passes `torch.Tensor` inputs through untouched, and returns a zero tensor fallback for `None` inside `numpy_to_pytorch()` (`experiments/phase8_4_master_validation_report.md` line 58).
- **What breaks without it:** Calling `x.astype(np.float32)` raises `AttributeError: 'dict' object has no attribute 'astype'` when WebUI API / IP-Adapter pipelines forward wrapped image dicts.

---

## Launch Arguments Note (`webui-user.bat` / `webui-user.sh`)
The reference workstation also sets `COMMANDLINE_ARGS=--api --xformers` in `webui/webui-user.bat` (`docs_generated/00_environment_and_run_log.md` also documents `--cuda-malloc`). Because launch flags are environment configuration rather than upstream code fixes, configure `--api --xformers` manually per OS guide.

## Clean-Copy Verification Status (`D:\5TH SEM\_forge_clean_test\`)
- **Unpatched Stock (`dfdcbab6...`) on `scene 1, seed 11` (`v2 default` `IP-Adapter Plus-Face SDXL`):** Completed without crashing (`RC=0`, `3.65 min`; no traceback in `forge_stock.log`; noted initial fallback to `animagine-xl-3.1` because empty `cache/` had not hashed Juggernaut-XL `[c9e3e68f89]`). Because the documented `IPAdapterPlus.py` / `utils.py` errors triggered historically on `FaceID Plus v2` / multi-unit / dict API workloads, the no-patch failure was **not reproduced** on `IP-Adapter Plus-Face SDXL`, so these `3` patches are shipped as **recommended**.
- **Patched Copy (`tools/apply_forge_patches.py --apply`) with Juggernaut-XL (`--ckpt ...`):** Applied cleanly (`check` exit `0`), `scene 1, seed 11` completed cleanly (`RC=0`, `3.69 min` render / `274.15s` total), `C:\Stable-Diffusion\webui` stayed untouched.

