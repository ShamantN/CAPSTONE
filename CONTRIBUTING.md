# Contributing to MMCCP v2.0

Welcome! Before setting up your environment or making changes, start here:
- **Setup & First Run:** [`docs/run/00_START_HERE.md`](docs/run/00_START_HERE.md) -> [`docs/run/01_NEW_MACHINE_WINDOWS.md`](docs/run/01_NEW_MACHINE_WINDOWS.md)
- **Licenses & Weights Policy:** [`docs/run/LICENSES_AND_MODELS.md`](docs/run/LICENSES_AND_MODELS.md)

---

## 1. Branch & Pull Request Workflow

- **Never push directly to `main` or `consistency-v2`.**
- Create a descriptive topic branch from updated `origin/main` (e.g. `feat/...`, `fix/...`, `docs/...`) and open a Pull Request targeting `main`.
- Wait for `@ShamantN` (Code Owner) approval and green checks before merging.

---

## 2. Non-Negotiable Scientific & Safety Invariants

### A. Frozen Verifier `v1.2` (`consistency_v2/verify/`)
- **NEVER edit files under [`consistency_v2/verify/`](consistency_v2/verify/) or [`consistency_v2/verifier_freeze.json`](consistency_v2/verifier_freeze.json)** in normal feature, pipeline, or docs PRs.
- Any modification breaks `tools/check_freeze.py` (`SHA: 3fa69071f888643b...`) and invalidates comparability across experiments.
- **Formal Verifier Change Protocol** *(only when opening a dedicated versioned verifier upgrade)*:
  1. Bump verifier version (e.g. `v1.2` -> `v1.3`).
  2. Pass all `15` control tests in [`tests/verifier_controls/test_controls.py`](tests/verifier_controls/test_controls.py).
  3. Re-score **all** historical and benchmark image sets with the new verifier version so comparisons remain apples-to-apples.
  4. Document every rule/threshold change in [`CHANGELOG.md`](CHANGELOG.md) and update freeze manifest hashes cleanly.
  5. Obtain explicit maintainer (`@ShamantN`) review & approval.

### B. Data Split & Seed Quarantine Rules (`consistency_v2/splits.py`)
- **`TEST-2` (`seeds 1101..1505` + `storyboards/holdout_v2.json`) is sealed/burned:** **Never** run tuning loops, ablation searches, or prompt/hyperparameter iteration against `TEST-2`, and **never** set `ALLOW_TEST2=1` during development.
- **Quarantined LoRA Seeds (`{606, 707, 808}`):** Permanently reserved and untouched across all tuning/dev scripts (`tools/audit_test2_isolation.py` rejects any literal reference).
- **Use DEV Splits Only:** Run experiments exclusively on **`DEV-2` (`seeds 101–505`)**, **`DEV-FAST` (`scenes 1,2,4,5 x seeds 11,23,37`)**, or **`seed 11`** smoke renders.

### C. Zero Manual Numbers in Documentation
- **Never hand-type or edit metric numbers** in benchmark summaries (`README_V2.md`, `docs_generated/v2/*.md`).
- Always regenerate [`results/numbers.json`](results/numbers.json) and rendered docs via:
  ```powershell
  python tools\collect_numbers.py
  python tools\render_docs.py
  python -m unittest tests\test_docs_numbers.py
  ```

### D. Repository & Secret Hygiene
- **Never commit** model weights (`.safetensors`, `.ckpt`, `.onnx`, `.pt`, `.pth`, `.bin`, `.h5`), raw bulk PNG render directories (`edu_outputs/`, `results/**/renders/`), `.env` files, or API keys/tokens.
- Use environment variables (`HF_TOKEN`, `GEMINI_API_KEY`, etc.) documented in [`.env.example`](.env.example).

---

## 3. Pre-PR Gate Checklist (Every PR Must Pass)

Run all checks locally before pushing:

```powershell
python tools\check_freeze.py
python tools\audit_test2_isolation.py
python tools\secret_scan.py
python -m unittest discover tests
python -m unittest tests\test_docs_numbers.py
```

*(Note: on a fresh clone, run `python tools\sync_canonical_banks.py` once first so the `7` bank/identity unit tests pass, making the full `unittest discover` suite pass.)*

---

## 4. Starter Tasks for Collaborators (`v2.1` Roadmap — Not Started in Frozen `v2.0`)

Two known architectural limits discovered during Phase R10 evaluation remain open for clean `v2.1` exploration on **DEV splits**:

1. **3+ Character Layout Partitioning (`M2 / M3 / M5`):**
   - *Current limit:* [`layout_planner.py`](consistency_v2/layout_planner.py), [`regional.py`](consistency_v2/regional.py), `prompt_builder.py`, and M5 face swap currently cap at **2 characters per scene**. In multi-character scenes expecting 3 people (e.g. Holdout Scene 4), the 3rd character is dropped (`Spatial 87.5%`).
   - *Goal:* Generalize column/mask spatial partitioning and `inswapper_128` zone binding to support `N >= 3` characters cleanly without OOM on 8 GB GPUs.
2. **Scene-State Wardrobe Override vs Static Anchor Wardrobe (`M7 WardrobeLock`):**
   - *Current limit:* [`wardrobe.py`](consistency_v2/wardrobe.py) locks each character to one static CIELAB clothing vector (`Delta E <= 38.0`) derived from their studio anchor portrait. When a storyboard explicitly describes an outfit change in a later scene (e.g. Ebright Scene 6 Graduation gown), M7 pulls clothing color back toward the original anchor shirt.
   - *Goal:* Parse scene-level outfit override descriptors from storyboard specs so M7 enforces scene-state wardrobe targets when an intentional costume change is specified, while retaining recurring wardrobe locks elsewhere.

