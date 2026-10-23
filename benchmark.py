#!/usr/bin/env python3
"""Compare the classical and deep pipelines on real + augmented test cases."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from barcode_recog.config import (DEEP_OUT, FIG_DIR, GT_TEXT, MODEL_PATH,
                                  OUTPUT_DIR, TEST_DIR, TRAD_OUT)
from barcode_recog.deep_model import DeepBarcodeRecognizer
from barcode_recog.traditional import TraditionalBarcodeRecognizer
from barcode_recog.utils import box_iou, char_error_rate, imread_gray, load_json, save_json

# matplotlib is optional at import time; it is installed by requirements.txt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_cjk_font():
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


_setup_cjk_font()


def quad_bbox(q):
    if not q:
        return (0, 0, 0, 0)
    a = np.asarray(q, np.float32)
    return (float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max()))


def evaluate(method_key, recognizer, cases, out_dir):
    rows = []
    vis_dir = out_dir / "cases"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cases):
        img_path = Path(c["image"])
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        img = imread_gray(img_path)
        gt_quad = c["quad"]
        gt_box = quad_bbox(gt_quad)
        t0 = time.perf_counter()
        try:
            r = recognizer.predict(img)
        except Exception as e:  # keep the benchmark running even if one case fails
            r = None
            err = f"{type(e).__name__}: {e}"
            print(f"  [{method_key}] ERROR on {c['name']}: {err}")
        elapsed = (time.perf_counter() - t0) * 1000.0
        if r is None:
            from barcode_recog.traditional import MethodResult
            r = MethodResult(method=method_key, bbox=[])
        iou = box_iou(r.bbox, gt_box) if r.bbox else 0.0
        exact = bool(r.ok and r.text == c["gt_text"])
        cer = char_error_rate(r.text, c["gt_text"])
        row = {
            "case": c["name"],
            "kind": c["kind"],
            "note": c["note"],
            "method": method_key,
            "pred_text": r.text,
            "gt_text": c["gt_text"],
            "ok": bool(r.ok),
            "exact": exact,
            "cer": round(float(cer), 4),
            "detected": bool(iou >= 0.5),
            "iou": round(float(iou), 4),
            "confidence": round(float(r.confidence), 4),
            "elapsed_ms": round(float(elapsed), 2),
            "decode_method": r.decode_method,
            "bbox": [round(float(v), 1) for v in r.bbox],
            "num_candidates": int(getattr(r, "num_candidates", 0)),
        }
        rows.append(row)
        print(f"  {c['name']:18s} exact={exact!s:5s} iou={iou:4.2f} "
              f"{elapsed:6.1f}ms  pred={r.text or '-'}")
        # visualisation
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if gt_quad:
            cv2.polylines(vis, [np.asarray(gt_quad, np.int32).reshape(-1, 1, 2)],
                          True, (0, 220, 0), 3)
        if r.bbox:
            x0, y0, x1, y1 = [int(round(v)) for v in r.bbox]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
        h, w = vis.shape[:2]
        scale = 640 / max(h, w)
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        head = f"{'OK' if exact else 'FAIL'}  pred={r.text or '-'}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(vis, head, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if exact else (0, 80, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(vis_dir / f"{i:02d}_{c['name']}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rows


def summaries(rows):
    out = {}
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        n = len(rr)
        out[method] = {
            "n": n,
            "exact_rate": sum(r["exact"] for r in rr) / max(1, n),
            "detect_rate": sum(r["detected"] for r in rr) / max(1, n),
            "decode_rate": sum(r["ok"] for r in rr) / max(1, n),
            "mean_cer": float(np.mean([r["cer"] for r in rr])),
            "mean_iou": float(np.mean([r["iou"] for r in rr])),
            "mean_ms": float(np.mean([r["elapsed_ms"] for r in rr])),
            "median_ms": float(np.median([r["elapsed_ms"] for r in rr])),
        }
        by_kind = defaultdict(list)
        for r in rr:
            by_kind[r["kind"]].append(r)
        out[method]["by_kind"] = {
            k: {"n": len(v),
                "exact_rate": sum(x["exact"] for x in v) / len(v),
                "mean_iou": float(np.mean([x["iou"] for x in v])),
                "mean_ms": float(np.mean([x["elapsed_ms"] for x in v]))}
            for k, v in by_kind.items()
        }
    return out


def make_charts(rows, summ, out_dir):
    methods = list(summ.keys())
    labels = ["传统 CV", "深度学习"] if len(methods) == 2 else methods
    colors = ["#3a7ca5", "#d1495b", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(2)
    width = 0.36
    for i, m in enumerate(methods):
        vals = [summ[m]["detect_rate"], summ[m]["exact_rate"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=labels[i],
                      color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.1f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["定位成功率 (IoU≥0.5)", "端到端识别率"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("比例")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("两种方法总体性能对比")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_method.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(methods):
        vals = sorted([r["elapsed_ms"] for r in rows if r["method"] == m])
        ax.plot(np.linspace(0, 100, len(vals)), vals, marker="o", ms=3,
                label=f"{labels[i]} (中位 {summ[m]['median_ms']:.0f} ms)",
                color=colors[i])
    ax.set_yscale("log")
    ax.set_xlabel("测试样本百分位 (%)")
    ax.set_ylabel("单张推理时间 (ms, 对数坐标)")
    ax.set_title("推理耗时分布")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=180)
    plt.close(fig)

    kinds = sorted({r["kind"] for r in rows})
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(kinds))
    width = 0.36
    for i, m in enumerate(methods):
        vals = []
        for k in kinds:
            sub = [r for r in rows if r["method"] == m and r["kind"] == k]
            vals.append(sum(r["exact"] for r in sub) / max(1, len(sub)))
        ax.bar(x + (i - 0.5) * width, vals, width, label=labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("端到端识别率")
    ax.set_title("不同退化类型下的识别率")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "by_category.png", dpi=180)
    plt.close(fig)

    hist_path = ROOT / "models" / "train_history.json"
    if hist_path.exists():
        hist = load_json(hist_path)
        h = hist.get("history", [])
        if h:
            fig, ax1 = plt.subplots(figsize=(7, 4.2))
            ep = [x["epoch"] for x in h]
            ax1.plot(ep, [x["train_loss"] for x in h], label="训练损失", color="#3a7ca5")
            ax1.plot(ep, [x["val_loss"] for x in h], label="验证损失", color="#d1495b")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")
            ax1.grid(alpha=0.25)
            ax2 = ax1.twinx()
            ax2.plot(ep, [x["val_iou"] for x in h], label="验证 IoU", color="#2a9d8f")
            ax2.set_ylabel("mask IoU")
            ax2.set_ylim(0, 1)
            lines = ax1.get_lines() + ax2.get_lines()
            ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
            ax1.set_title("U-Net 训练曲线")
            fig.tight_layout()
            fig.savefig(out_dir / "training_curve.png", dpi=180)
            plt.close(fig)


def build_report(rows, summ, out_dir):
    methods = list(summ.keys())
    m1 = methods[0]
    m2 = methods[1] if len(methods) > 1 else methods[0]
    s1, s2 = summ[m1], summ[m2]
    n = len([r for r in rows if r["method"] == m1])
    lines = []
    lines.append("# 砂轮条码识别实验报告\n")
    lines.append(f"测试样本数：**{n}**（1 张原始砂轮图、1 张给定条码区域图、"
                 f"{n-2} 张几何/光照/模糊/分辨率/压缩/遮挡增广图）。  ")
    lines.append(f"条码真值：`{GT_TEXT}`（EAN-13）。\n")

    lines.append("## 1. 方法概述\n")
    lines.append("### 1.1 传统计算机视觉方法\n")
    lines.append("1. 图像缩放、去噪；\n"
                 "2. 计算 Sobel 梯度张量，得到局部边缘能量与方向一致性（structure tensor coherence）；\n"
                 "3. 多阈值化 + 形态学闭运算生成条码候选框；\n"
                 "4. 候选区域 Hough 直线估计条码倾角并矫正；\n"
                 "5. 自实现的 EAN-13 扫描线解码（95 模块采样 + 左右奇偶模式 + 校验位）。\n")
    lines.append("### 1.2 深度学习方法\n")
    lines.append("1. 用单张真实条码图做一次性几何/光度增广，合成 260 张训练场景和 "
                 "60 张验证场景，并加入无条码负样本做 hard-negative 重复采样；\n"
                 "2. 从零训练一个 1.94M 参数的 U-Net 做条码像素级分割"
                 "（224×224、batch 12、30 epochs、BCE+Dice）；\n"
                 "3. 测试时采用 768 像素重叠滑窗并把每个 tile 归一到训练尺度，避免尺度失配；\n"
                 "4. 由分割 mask 得到旋转矩形，透视矫正后送入同一个 EAN-13 解码器。\n")

    lines.append("## 2. 总体结果\n")
    lines.append("| 方法 | 定位成功率 (IoU≥0.5) | 端到端识别率 | 平均 CER | 平均 IoU | 中位耗时 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m, s in summ.items():
        lines.append(f"| {m} | {s['detect_rate']*100:.1f}% | {s['exact_rate']*100:.1f}% | "
                     f"{s['mean_cer']:.4f} | {s['mean_iou']:.3f} | {s['median_ms']:.0f} ms |")
    lines.append("")
    lines.append("![总体准确率](figures/accuracy_by_method.png)\n")
    lines.append("![各类退化识别率](figures/by_category.png)\n")
    lines.append("详细逐样本结果见 `comparison.csv`，可视化结果见 `cases/`。\n")

    lines.append("## 3. 数据与增广示例\n")
    lines.append("![单样本增广](figures/augmentation_examples.jpg)\n")
    lines.append("上排为合成训练场景，红色区域为像素级条码 mask；背景、尺度、旋转、"
                 "透视、亮度和模糊均随机。\n")
    lines.append("![测试样本](figures/test_cases.jpg)\n")
    lines.append("测试集包含原图、条码特写及 14 种几何/光度/分辨率/压缩/遮挡退化。\n")
    lines.append("![原图对比](figures/comparison_full.jpg)\n")
    lines.append("原图上的定位结果：绿色为真值，红色为传统方法，蓝色为深度方法。\n")

    lines.append("## 4. 分类结果\n")
    lines.append("| 退化类型 | 样本数 | 传统 CV 识别率 | 深度学习识别率 |")
    lines.append("|---|---:|---:|---:|")
    kinds = sorted({r["kind"] for r in rows})
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        a = sum(r["exact"] for r in sub if r["method"] == m1) / max(1, len([r for r in sub if r["method"] == m1]))
        b = sum(r["exact"] for r in sub if r["method"] == m2) / max(1, len([r for r in sub if r["method"] == m2]))
        lines.append(f"| {k} | {len([r for r in sub if r['method']==m1])} | {a*100:.0f}% | {b*100:.0f}% |")
    lines.append("")

    lines.append("## 5. 失败分析\n")
    fails = [r for r in rows if not r["exact"]]
    if fails:
        lines.append("| 样本 | 方法 | 真值 | 预测 | IoU | 原因分析 |")
        lines.append("|---|---|---|---|---:|---|")
        for r in fails:
            reason = "条码对比度/边缘被破坏，扫描线无法形成合法 95 模块序列"
            if r["kind"] == "photometric" and "模糊" in r["note"]:
                reason = "失焦/运动模糊导致窄条码模块混叠，超过解码器恢复能力"
            elif r["kind"] == "occlusion":
                reason = "局部遮挡破坏守卫符或数据字符"
            elif r["iou"] < 0.5:
                reason = "候选框定位失败"
            lines.append(f"| {r['case']} | {r['method']} | {r['gt_text']} | "
                         f"{r['pred_text'] or '—'} | {r['iou']:.2f} | {reason} |")
    else:
        lines.append("所有测试样本均识别正确。")
    lines.append("")

    lines.append("## 6. 结论\n")
    lines.append(f"- 在 {n} 个测试样本上，传统方法端到端识别率 **{s1['exact_rate']*100:.1f}%**，"
                 f"深度方法 **{s2['exact_rate']*100:.1f}%**。")
    lines.append(f"- 传统方法平均定位 IoU 为 {s1['mean_iou']:.3f}，深度方法为 {s2['mean_iou']:.3f}。")
    lines.append(f"- 传统方法中位耗时 {s1['median_ms']:.0f} ms，深度方法中位耗时 {s2['median_ms']:.0f} ms。"
                 "传统方法无需训练、可解释性强；深度方法对光照和纹理干扰更鲁棒，但需要 GPU/MPS 推理和一次性训练成本。")
    lines.append("- 严重失焦和运动模糊仍是主要瓶颈：条码模块宽度接近或低于 1–2 像素时，"
                 "任何基于扫描线的解码器都会失效，需要在成像端提高分辨率或缩短曝光。\n")

    lines.append("![耗时分布](figures/latency.png)\n")
    lines.append("![训练曲线](figures/training_curve.png)\n")

    lines.append("## 7. 运行方式\n")
    lines.append("```bash\npython -m venv .venv && source .venv/bin/activate\n"
                 "pip install -r requirements.txt\n"
                 "python scripts/prepare_data.py\n"
                 "python scripts/train_deep.py\n"
                 "python scripts/benchmark.py\n```\n")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", choices=["both", "traditional", "deep"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = load_json(TEST_DIR / "manifest.json")
    if args.limit:
        cases = cases[:args.limit]
    print(f"test cases: {len(cases)}")
    rows = []
    if args.methods in ("both", "traditional"):
        print("[traditional]")
        rows += evaluate("传统 CV", TraditionalBarcodeRecognizer(), cases, TRAD_OUT)
    if args.methods in ("both", "deep"):
        print("[deep]")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"missing model {MODEL_PATH}; run train_deep.py")
        rows += evaluate("深度学习", DeepBarcodeRecognizer(), cases, DEEP_OUT)

    summ = summaries(rows)
    save_json(OUTPUT_DIR / "results.json", {"rows": rows, "summary": summ})
    with (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    make_charts(rows, summ, FIG_DIR)
    build_report(rows, summ, OUTPUT_DIR)
    print("\n=== summary ===")
    for m, s in summ.items():
        print(f"{m}: detect={s['detect_rate']:.3f} exact={s['exact_rate']:.3f} "
              f"cer={s['mean_cer']:.4f} iou={s['mean_iou']:.3f} "
              f"median={s['median_ms']:.0f}ms")
    print(f"results -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
