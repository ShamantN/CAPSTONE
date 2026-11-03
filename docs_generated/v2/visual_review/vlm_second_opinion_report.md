# Phase R8: Local VLM Second Opinion Review Report

**Date:** 2026-11-02  
**Backbones:** `Salesforce/blip-image-captioning-base` & `openai/clip-vit-base-patch32` (Pinned to CPU)  
**Inter-Rater Agreement on Person Count (Cohen's Kappa):** **0.1220** (Moderate Agreement)  
**Inter-Rater Agreement on Wardrobe (Cohen's Kappa):** **-0.1250**  
**Inter-Rater Agreement on Overall Pass (Cohen's Kappa):** **-0.0500**  

## Per-Scene VLM Perception Summary

| File | Scene | Expected | VLM Count | YOLO Count | BLIP Caption | VLM Pass | Verifier Pass |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| `scene_1_seed_101.png` | 1 | 2 | 2 | 2 | "a woman and a child are sitting at a table" | PASS | PASS |
| `scene_1_seed_202.png` | 1 | 2 | 2 | 2 | "a woman and a child sitting at a table" | PASS | PASS |
| `scene_2_seed_101.png` | 2 | 1 | 2 | 1 | "a young boy sitting on a table with a yellow box" | FAIL | PASS |
| `scene_2_seed_202.png` | 2 | 1 | 2 | 1 | "a young boy in a yellow shirt and blue hat" | FAIL | PASS |
| `scene_3_seed_101.png` | 3 | 1 | 1 | 1 | "a boy sitting at a table with a piece of paper" | PASS | PASS |
| `scene_3_seed_202.png` | 3 | 1 | 2 | 1 | "a boy sitting on a bed" | FAIL | PASS |
| `scene_4_seed_101.png` | 4 | 1 | 2 | 1 | "a young boy leaning on a fence in a garden" | FAIL | PASS |
| `scene_4_seed_202.png` | 4 | 1 | 2 | 1 | "a boy standing in front of a box of food" | FAIL | PASS |
| `scene_5_seed_101.png` | 5 | 2 | 3 | 2 | "a man sitting on a chair" | FAIL | PASS |
| `scene_5_seed_202.png` | 5 | 2 | 2 | 2 | "a young boy in a hallway" | PASS | FAIL |
| `scene_6_seed_101.png` | 6 | 1 | 1 | 1 | "a man in a graduation gown holding a red book" | PASS | PASS |
| `scene_6_seed_202.png` | 6 | 1 | 2 | 5 | "a man in a graduation gown holding a red book" | FAIL | FAIL |
