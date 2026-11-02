# Phase R9: Clean-Slate Zero-Cache Telemetry & Latency Report

**Date:** 2026-11-02  
**Whole-System Peak VRAM Observed:** **7964.0 MB** (7.777 GB)  
**Forge Dedicated GPU Allocation Cap:** **3200.0 MB** (`xl_GPU_MB`) + 1024 MB (`forge_inference_memory`)  
**Host RAM Stability:** Peak **28772.9 MB** / 32768.0 MB (Zero paging / OOM events)  
**Telemetry Duration:** 12725 seconds (1 Hz continuous sampling)  

## Execution Latency Comparison (Cold-Start Zero Cache)

| Pipeline / Mode | Scene 1 | Scene 2 | Scene 3 | Scene 4 | Scene 5 | Scene 6 | Total Time | Mean / Scene |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v2_default` | 130.8s | 79.5s | 79.5s | 77.3s | 126.5s | 68.7s | **9.37m** | 93.7s |
| `v2_quality` | 149.5s | 79.5s | 80.2s | 81.1s | 141.6s | 154.4s | **11.44m** | 114.4s |
| `v1_baseline` | 88.3s | 91.5s | 97.2s | 100.0s | 102.0s | 89.9s | **9.48m** | 94.8s |
