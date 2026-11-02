# Phase R8: Local VLM Second Opinion Review Report

**Date:** 2026-11-02  
**Backbones:** `Salesforce/blip-image-captioning-base` & `openai/clip-vit-base-patch32` (Pinned to CPU)  
**Inter-Rater Agreement on Person Count (Cohen's Kappa):** **0.6250** (Substantial Agreement)  
**Inter-Rater Agreement on Wardrobe (Cohen's Kappa):** **0.2857**  
**Inter-Rater Agreement on Overall Pass (Cohen's Kappa):** **0.4706**  

## Per-Scene VLM Perception Summary

| File | Scene | Expected | VLM Count | YOLO Count | BLIP Caption | VLM Pass | Verifier Pass |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| `scene_1_seed_101.png` | 1 | 2 | 2 | 2 | "a woman and a child are sitting at a table" | PASS | PASS |
| `scene_1_seed_202.png` | 1 | 2 | 2 | 2 | "a woman sitting at a table with a child" | PASS | PASS |
| `scene_2_seed_101.png` | 2 | 1 | 2 | 2 | "a young boy sitting on a bench with a bunch of butterflies flying around him" | FAIL | FAIL |
| `scene_2_seed_202.png` | 2 | 1 | 1 | 2 | "a young boy standing in front of a butterfly cage" | PASS | FAIL |
| `scene_3_seed_101.png` | 3 | 1 | 1 | 1 | "a man sitting on a bed" | PASS | PASS |
| `scene_3_seed_202.png` | 3 | 1 | 1 | 1 | "a young boy sitting on a stool in a room" | PASS | PASS |
| `scene_4_seed_101.png` | 4 | 1 | 1 | 1 | "a young boy sitting on a fence in a garden" | PASS | PASS |
| `scene_4_seed_202.png` | 4 | 1 | 1 | 1 | "a young boy standing in front of a garden" | PASS | PASS |
| `scene_5_seed_101.png` | 5 | 2 | 4 | 4 | "a man in a yellow shirt" | FAIL | FAIL |
| `scene_5_seed_202.png` | 5 | 2 | 3 | 2 | "two young men sitting on a bench" | FAIL | PASS |
| `scene_6_seed_101.png` | 6 | 1 | 1 | 1 | "a man in a graduation gown holding a diploma" | FAIL | PASS |
| `scene_6_seed_202.png` | 6 | 1 | 2 | 3 | "a man in a graduation gown holding a red book" | FAIL | FAIL |
