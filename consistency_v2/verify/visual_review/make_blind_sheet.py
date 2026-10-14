"""
Blind Re-Identification Test Generator (VR-4)
Extracts face crops from multiple scenes, shuffles them with randomized IDs,
builds an anonymous visual test board, and saves a separate sealed answer key.
"""
import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


def generate_blind_reid_test(
    face_crops_with_labels: List[Tuple[str, np.ndarray]],  # list of (char_id, crop_bgr)
    output_image_path: Path,
    output_key_path: Path,
    grid_cols: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Shuffles face crops, lays them out on a grid numbered 1..N,
    and writes answer key to output_key_path.
    """
    rng = random.Random(seed)
    indexed_crops = list(enumerate(face_crops_with_labels))
    rng.shuffle(indexed_crops)

    answer_key = {}
    tiles = []
    tile_size = 180

    for new_idx, (orig_idx, (char_id, crop_bgr)) in enumerate(indexed_crops, start=1):
        test_id = f"Face_{new_idx:02d}"
        answer_key[test_id] = {
            "true_character_id": char_id,
            "original_index": orig_idx
        }

        # Format crop
        crop_resized = cv2.resize(crop_bgr, (tile_size, tile_size))
        # Add border and label
        tile = np.zeros((tile_size + 40, tile_size, 3), dtype=np.uint8)
        tile[:] = (30, 32, 38)
        tile[:tile_size, :] = crop_resized
        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), (80, 85, 95), 1)

        # Draw number
        cv2.putText(tile, f"#{new_idx:02d}", (15, tile_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 250), 2, cv2.LINE_AA)
        tiles.append(tile)

    # Grid canvas
    n_tiles = len(tiles)
    grid_rows = (n_tiles + grid_cols - 1) // grid_cols
    total_w = grid_cols * (tile_size + 20) + 20
    total_h = grid_rows * (tile_size + 60) + 100

    board = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    board[:] = (20, 22, 26)

    # Title
    cv2.putText(board, "MMCCP v2 Blind Character Re-Identification Test Board", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 240), 2, cv2.LINE_AA)
    cv2.putText(board, "Identify character identity for each anonymous numbered crop without opening answer key.", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 165, 175), 1, cv2.LINE_AA)

    for i, tile in enumerate(tiles):
        r = i // grid_cols
        c = i % grid_cols
        tx = 20 + c * (tile_size + 20)
        ty = 100 + r * (tile_size + 60)
        th, tw = tile.shape[:2]
        board[ty:ty + th, tx:tx + tw] = tile

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_key_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_image_path), board)
    with open(output_key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2)

    return answer_key
