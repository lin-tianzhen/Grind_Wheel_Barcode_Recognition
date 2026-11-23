"""深度学习砂轮条形码检测识别.

技术路线:
  1. U-Net 语义分割网络定位条码区域 (输出像素级 mask)
  2. 损失函数: BCEWithLogitsLoss + DiceLoss
  3. 优化器: AdamW + CosineAnnealingLR
  4. 推理: 滑窗推理 → 连通域 → 最小外接矩形 → 透视矫正 → EAN-13 解码

具体实现已拆分到 Unet_model/ 包:
  config / model / losses / dataset / train / decode / inference / evaluate / main

用法:
  python main.py                          # 训练 (默认200 epochs)
  python main.py --epochs 50 --size 224   # 自定义参数
  python main.py --eval-only              # 仅评估已训练模型
  python main.py --infer data/raw/RegulaImg.bmp  # 单张推理
  python main.py --infer-dir data/raw     # 批量推理目录下所有图片
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import torch

from config import MODEL_DIR, OUTPUT_DIR, imread_unicode
from dataset import build_real_dataset
from evaluate import evaluate, visualize_prediction
from inference import infer_single
from model import UNet
from train import get_device, train

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_trained_model(device, base_ch: int):
    """加载已训练的 U-Net, 返回 (model, train_size, ckpt); 模型不存在返回 (None, None, None)."""
    model_path = MODEL_DIR / "unet_barcode.pt"
    if not model_path.exists():
        return None, None, None
    model = UNet(in_ch=1, base_ch=base_ch, dropout=0.0).to(device)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
    return model, train_size, ckpt


def infer_directory(model, dir_path: Path, device, train_size: int,
                    long_side: int, use_fallback: bool):
    """对目录下所有图片推理, 逐张输出检测到的条码值 (真值未知时以解码成功为准)."""
    files = sorted(p for p in dir_path.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"[警告] 目录中没有可识别的图片: {dir_path}")
        return
    print(f"目录: {dir_path} 共 {len(files)} 张图片")
    print("-" * 60)
    ok_count = 0
    total_ms = 0.0
    failed = []
    codes: dict = {}
    for p in files:
        res = infer_single(model, p, device, train_size=train_size,
                           long_side=long_side, use_fallback=use_fallback,
                           gt_text=None)
        elapsed_ms = res.get("elapsed_ms", 0.0)
        total_ms += elapsed_ms
        text = res.get("text") or ""
        if text:
            checksum = "校验通过" if res["checksum_ok"] else "校验未通过"
        else:
            checksum = "未解出"
        method = res.get("method") or "-"
        err = res.get("error")
        mark = "✓" if res["ok"] else "✗"
        print(f"  {mark} {p.name}: {text or '—'} "
              f"({checksum}, {elapsed_ms:.0f}ms, {method})"
              f"{f' [{err}]' if err else ''}")
        if res["ok"]:
            ok_count += 1
            codes[text] = codes.get(text, 0) + 1
        else:
            failed.append(p.name)
    print("-" * 60)
    print(f"成功解码: {ok_count}/{len(files)}")
    print(f"平均耗时: {total_ms / len(files):.0f} ms/张")
    if codes:
        print("检测到的条码值分布:")
        for text, count in sorted(codes.items(), key=lambda kv: -kv[1]):
            print(f"  {text}: {count} 张")
    if failed:
        print(f"未成功解码 ({len(failed)} 张): {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(
        description="深度学习砂轮条形码检测识别")
    parser.add_argument("--build-data", action="store_true",
                        help="从 data/raw 构建真实域定位数据集后退出")
    parser.add_argument("--data-pos", type=int, default=60,
                        help="每张源图生成的正样本数 (默认60)")
    parser.add_argument("--data-neg", type=int, default=18,
                        help="每张源图生成的负样本数 (默认18)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数 (默认200)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批大小 (默认16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率 (默认2e-4)")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="权重衰减 (默认1e-5)")
    parser.add_argument("--size", type=int, default=384,
                        help="训练图像尺寸 (默认384)")
    parser.add_argument("--base-ch", type=int, default=24,
                        help="U-Net 基础通道数 (默认24)")
    parser.add_argument("--pos-weight", type=float, default=4.0,
                        help="正样本权重 (默认4.0)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 率 (默认0.1)")
    parser.add_argument("--patience", type=int, default=40,
                        help="早停耐心轮数 (默认40)")
    parser.add_argument("--long-side", type=int, default=1280,
                        help="推理定位时的图像长边 (默认1280)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="关闭传统方法兜底 (仅评估纯 U-Net 效果)")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已训练模型")
    parser.add_argument("--infer", type=str, default=None,
                        help="对单张图像推理")
    parser.add_argument("--infer-dir", type=str, default=None,
                        help="对指定目录下所有图片批量推理 (输出检测到的条码值)")
    args = parser.parse_args()

    print("=" * 60)
    print("  深度学习砂轮条形码检测识别")
    print("  U-Net 分割 + 批量滑窗推理 + EAN-13 解码")
    print("=" * 60)

    device = get_device()
    print(f"设备: {device}\n")

    if args.build_data:
        build_real_dataset(size=args.size, n_pos=args.data_pos,
                           n_neg=args.data_neg)
        return

    use_fallback = not args.no_fallback

    if args.infer_dir:
        dir_path = Path(args.infer_dir)
        if not dir_path.is_dir():
            print(f"[错误] 目录不存在: {dir_path}")
            return
        model, train_size, ckpt = load_trained_model(device, args.base_ch)
        if model is None:
            print(f"[错误] 模型文件不存在: {MODEL_DIR / 'unet_barcode.pt'}")
            print("请先运行训练: python main.py")
            return
        print(f"已加载模型: {MODEL_DIR / 'unet_barcode.pt'} "
              f"(epoch {ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}, "
              f"train_size={train_size})\n")
        infer_directory(model, dir_path, device, train_size,
                        args.long_side, use_fallback)
        return

    if args.infer:
        model, train_size, ckpt = load_trained_model(device, args.base_ch)
        if model is None:
            print(f"[错误] 模型文件不存在: {MODEL_DIR / 'unet_barcode.pt'}")
            print("请先运行训练: python main.py")
            return
        print(f"已加载模型: {MODEL_DIR / 'unet_barcode.pt'} (epoch {ckpt['epoch']}, "
              f"val_iou={ckpt['val_iou']:.4f}, train_size={train_size})")
        img_path = Path(args.infer)
        res = infer_single(model, img_path, device, train_size=train_size,
                           long_side=args.long_side, use_fallback=use_fallback)
        status = "✓ 成功" if res["ok"] else "✗ 失败"
        print(f"\n结果: {status}")
        print(f"条码值: {res['text'] or '—'}")
        print(f"校验位: {'通过' if res['checksum_ok'] else '未通过'}")
        print(f"定位方式: {res.get('method') or '—'}")
        print(f"耗时: {res['elapsed_ms']:.0f} ms")
        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        prob_map = res.get("prob_map")
        if img is not None and prob_map is not None:
            scale = res.get("scale", 1.0)
            img_r = (cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else img)
            vis_path = OUTPUT_DIR / f"{img_path.stem}_deep_result.jpg"
            visualize_prediction(img_r, prob_map, res, vis_path)
            print(f"可视化已保存: {vis_path}")
        return

    if args.eval_only:
        model, train_size, ckpt = load_trained_model(device, args.base_ch)
        if model is None:
            print(f"[错误] 模型文件不存在: {MODEL_DIR / 'unet_barcode.pt'}")
            return
        print(f"已加载模型: {MODEL_DIR / 'unet_barcode.pt'}, train_size={train_size}")
        evaluate(model, device, train_size=train_size,
                 long_side=args.long_side, use_fallback=use_fallback)
        return

    train(args)

    print("\n训练完成，正在评估...")
    model, train_size, ckpt = load_trained_model(device, args.base_ch)
    evaluate(model, device, train_size=train_size,
             long_side=args.long_side, use_fallback=use_fallback)


if __name__ == "__main__":
    main()