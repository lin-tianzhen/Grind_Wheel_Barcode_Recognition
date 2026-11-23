#!/usr/bin/env python3
"""训练历史可视化: 读取 models/train_history.json, 生成损失与 IoU 折线图。

用法:
    python plot_history.py                     # 使用默认路径, 保存到 outputs/
    python plot_history.py --show              # 生成后弹出窗口查看
    python plot_history.py --smooth 10         # 对曲线做 10 点移动平均平滑
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# 无图形界面环境下也能保存图片; 显式 --show 时使用交互后端
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_PATH = PROJECT_ROOT / "models" / "train_history.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _setup_chinese_font() -> None:
    """配置中文字体, 避免中文标签显示为方框。"""
    from matplotlib import font_manager
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """滑动窗口移动平均, 返回长度 = len(x) - window + 1 的序列。"""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _plot_curve(ax, epochs, values, label, color, smooth, marker_best,
                best_index, best_label):
    """在 ax 上绘制一条曲线, 可选平滑与最优 epoch 标注。"""
    if values.size == 0:
        return
    if smooth > 1 and values.size >= smooth:
        x_s = np.arange(smooth, values.size + 1)
        y_s = moving_average(values, smooth)
        ax.plot(x_s, y_s, label=f"{label} (平滑)", color=color, linewidth=1.6)
        ax.plot(epochs, values, color=color, linewidth=0.6, alpha=0.35)
    else:
        ax.plot(epochs, values, label=label, color=color, linewidth=1.2)

    if marker_best and best_index is not None:
        ax.scatter([best_index], [values[best_index - 1]], color="red",
                   zorder=5, s=40)
        ax.annotate(
            f"{best_label}: epoch {best_index}\n{values[best_index - 1]:.4f}",
            xy=(best_index, values[best_index - 1]),
            xytext=(8, 12), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练历史折线图可视化")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH,
                        help="train_history.json 路径(默认 models/train_history.json)")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出 PNG 路径(默认 outputs/training_history.png)")
    parser.add_argument("--smooth", type=int, default=0,
                        help="移动平均窗口大小, 0 表示不平滑(默认 0)")
    parser.add_argument("--show", action="store_true",
                        help="生成后弹出窗口显示")
    args = parser.parse_args()

    history_path = args.history
    if not history_path.exists():
        raise SystemExit(f"未找到历史文件: {history_path}")

    data = json.loads(history_path.read_text(encoding="utf-8"))

    train_loss = np.asarray(data.get("train_loss", []), dtype=np.float64)
    val_loss = np.asarray(data.get("val_loss", []), dtype=np.float64)
    val_iou = np.asarray(data.get("val_iou", []), dtype=np.float64)

    if train_loss.size == 0:
        raise SystemExit(f"{history_path} 中没有 train_loss 数据")

    _setup_chinese_font()

    epochs = np.arange(1, len(train_loss) + 1)
    best_loss_epoch = int(np.argmin(val_loss)) + 1 if val_loss.size else None
    best_iou_epoch = int(np.argmax(val_iou)) + 1 if val_iou.size else None

    # 三个指标用 2 个子图: 左侧损失, 右侧 IoU
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("U-Net 训练历史", fontsize=14)

    # ---- 左图: 损失曲线 ----
    ax = axes[0]
    _plot_curve(ax, epochs, train_loss, "训练损失 (train_loss)", "#1f77b4",
                args.smooth, marker_best=False, best_index=None, best_label="")
    _plot_curve(ax, epochs, val_loss, "验证损失 (val_loss)", "#ff7f0e",
                args.smooth, marker_best=True,
                best_index=best_loss_epoch, best_label="最优")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练 / 验证损失")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 右图: IoU 曲线 ----
    ax = axes[1]
    _plot_curve(ax, epochs, val_iou, "验证 IoU (val_iou)", "#2ca02c",
                args.smooth, marker_best=True,
                best_index=best_iou_epoch, best_label="最优")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("验证集 IoU")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = args.out or (OUTPUT_DIR / "training_history.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"折线图已保存至: {out_path}")
    print(f"  epochs = {len(train_loss)}, "
          f"最优 val_loss = {val_loss.min():.4f} (epoch {best_loss_epoch}), "
          f"最优 val_iou = {val_iou.max():.4f} (epoch {best_iou_epoch})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()