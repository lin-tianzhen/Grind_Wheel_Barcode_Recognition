#!/usr/bin/env python3
"""Recognise the EAN-13 barcode in one image with either/both methods."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from barcode_recog.config import OUTPUT_DIR
from barcode_recog.deep_model import DeepBarcodeRecognizer
from barcode_recog.traditional import TraditionalBarcodeRecognizer
from barcode_recog.utils import imread_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", choices=["both", "traditional", "deep"], default="both")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "single"))
    args = ap.parse_args()

    img = imread_gray(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if args.method in ("both", "traditional"):
        r = TraditionalBarcodeRecognizer(debug=True).predict(img)
        results.append(r.to_dict())
    if args.method in ("both", "deep"):
        r = DeepBarcodeRecognizer().predict(img)
        results.append(r.to_dict())

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r, color in zip(results, [(0, 0, 255), (255, 120, 0)]):
        if r["bbox"]:
            x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)
    h, w = vis.shape[:2]
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "annotated.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out_dir / "result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['method']}: ok={r['ok']} text={r['text']} "
              f"bbox={r['bbox']} time={r['elapsed_ms']:.0f}ms")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
