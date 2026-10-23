#!/usr/bin/env python3
"""Build the augmented training/validation set and the held-out test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from augment import build_synthetic_dataset, build_test_set
from config import (SEED, SYNTH_DIR, TEST_DIR, TRAIN_PER_IMAGE,
                                  VAL_PER_IMAGE, SCENE_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=TRAIN_PER_IMAGE)
    ap.add_argument("--val", type=int, default=VAL_PER_IMAGE)
    ap.add_argument("--size", type=int, default=SCENE_SIZE)
    ap.add_argument("--test-side", type=int, default=2000)
    args = ap.parse_args()

    print(f"[1/3] synthetic train scenes: {args.train} x {args.size}px ...")
    build_synthetic_dataset(SYNTH_DIR, args.train, args.size, SEED, 1.0, "train")
    print(f"[2/3] synthetic val scenes:   {args.val} x {args.size}px ...")
    build_synthetic_dataset(SYNTH_DIR, args.val, args.size, SEED + 1, 0.9, "val")
    print("[3/3] real + augmented test cases ...")
    cases = build_test_set(TEST_DIR, max_side=args.test_side, seed=SEED + 7)
    print(f"done: {args.train} train, {args.val} val, {len(cases)} test cases")


if __name__ == "__main__":
    main()
