## Summary
<!-- Briefly describe what this PR changes and why. -->

## Invariant & Gate Checklist (Required Before Review)
- [ ] **Frozen Verifier Intact:** `python tools/check_freeze.py` exits `0` (`consistency_v2/verify/` and `consistency_v2/verifier_freeze.json` unmodified)
- [ ] **TEST-2 & Reserved Seed Isolation:** `python tools/audit_test2_isolation.py` exits `0` (no tuning on `TEST-2`, no `ALLOW_TEST2=1` in dev, no reserved seeds `{606, 707, 808}`)
- [ ] **Secret Scan Clean:** `python tools/secret_scan.py` reports `CLEAN` (zero `.env`, API tokens, or hardcoded credentials)
- [ ] **Unit & Doc-Lint Tests Passing:** `python -m unittest discover tests` and `python -m unittest tests/test_docs_numbers.py` pass locally
- [ ] **DEV-Only Tuning:** Any new generation/hyperparameter experiments ran strictly on `DEV` splits (`DEV-2` / `DEV-FAST` / `seed 11`)
- [ ] **No Weights or Render Bloat Committed:** Verified zero `.safetensors` / `.onnx` / `.pth` / `.h5` / `.bin` weights or raw render PNG folders staged
- [ ] **Canonical Docs Script-Generated (if metrics changed):** Re-ran `python tools/collect_numbers.py` and `python tools/render_docs.py` instead of hand-editing numbers
