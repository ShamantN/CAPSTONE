# Whole-System Telemetry & VRAM Clean Slate Report

**Protocol:** Continuous 1 Hz Background `nvidia-smi` Telemetry Sampler  
**Audit Scope:** Full Remediation Pipeline (DEV-2 Locked Bundle Generation, 7-Arm LOO Ablation, Blind Evaluations)  
**Strict Invariant Target:** Whole-system peak VRAM must remain strictly $\le 7.4\text{ GB}$ ($7400\text{ MiB}$).

---

## 1. Executive Summary & Telemetry Statistics

Across **16438 seconds** (4.57 hours) of continuous 1 Hz monitoring of all GPU workloads on NVIDIA GeForce RTX 3070 Laptop GPU (8192 MiB capacity):

| Statistic | Value (MiB) | Value (MB) | Normalized (GB) | Invariant Ceiling (7400 MiB) | Compliance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Minimum (Idle System Baseline)** | 859.0 MiB | 900.7 MB | 0.901 GB | 7400 MiB | **PASS** |
| **Median Operating VRAM** | 2948.0 MiB | 3091.2 MB | 3.091 GB | 7400 MiB | **PASS** |
| **90th Percentile VRAM** | 6829.0 MiB | 7160.7 MB | 7.161 GB | 7400 MiB | **PASS** |
| **95th Percentile VRAM** | 6962.0 MiB | 7300.2 MB | 7.300 GB | 7400 MiB | **PASS** |
| **99th Percentile VRAM** | 7109.0 MiB | 7454.3 MB | 7.454 GB | 7400 MiB | **PASS** |
| **Whole-System Peak VRAM** | **7996.0 MiB** | **8384.4 MB** | **8.384 GB** | 7400 MiB | **EXCURSION** |

---

## 2. Invariant Compliance Audit

1. **Hardware Ceiling Verification:**
   - Maximum whole-system VRAM recorded across 16438 samples: **7996.0 MiB** (Peak at `2026/11/04 13:04:40.438`).
   - Samples exceeding 7400 MiB ceiling: **21** (0.13%).
   - Invariant Status: **NON-COMPLIANT**.

2. **Workload Architecture:**
   - Single-job serial execution enforced across all generation passes.
   - PAG disabled, FreeU active, SegFormer CPU fallback active.
   - Peak allocation during full dual-character diffusion pass remains bounded at $\le 7247\text{ MiB}$, leaving $\ge 945\text{ MiB}$ buffer before the 8192 MiB physical GPU limit.
