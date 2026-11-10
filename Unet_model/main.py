"""深度学习砂轮条形码检测识别.

技术路线:
  1. U-Net 语义分割网络定位条码区域 (输出像素级 mask)
  2. 损失函数: BCEWithLogitsLoss + DiceLoss
  3. 优化器: AdamW + CosineAnnealingLR
  4. 推理: 滑窗推理 → 连通域 → 最小外接矩形 → 透视矫正 → EAN-13 解码

具体实现已拆分到 Unet_model/ 包:
  config / model / losses / dataset / train / decode / inference / evaluate / main

用法:
  python deep_model_UNet.py                          # 训练 (默认200 epochs)
  python deep_model_UNet.py --epochs 50 --size 224   # 自定义参数
  python deep_model_UNet.py --eval-only              # 仅评估已训练模型
  python deep_model_UNet.py --infer data/raw/RegulaImg.bmp  # 单张推理
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from .config import MODEL_DIR, OUTPUT_DIR, imread_unicode
from .dataset import build_real_dataset
from .evaluate import evaluate, visualize_prediction
from .inference import infer_single
from .model import UNet
from .train import get_device, train


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

    if args.infer:
        model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
        model_path = MODEL_DIR / "unet_barcode.pt"
        if not model_path.exists():
            print(f"[错误] 模型文件不存在: {model_path}")
            print("请先运行训练: python deep_model_UNet.py")
            return
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
        print(f"已加载模型: {model_path} (epoch {ckpt['epoch']}, "
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
        model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
        model_path = MODEL_DIR / "unet_barcode.pt"
        if not model_path.exists():
            print(f"[错误] 模型文件不存在: {model_path}")
            return
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
        print(f"已加载模型: {model_path}, train_size={train_size}")
        evaluate(model, device, train_size=train_size,
                 long_side=args.long_side, use_fallback=use_fallback)
        return

    train(args)

    print("\n训练完成，正在评估...")
    model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
    ckpt = torch.load(str(MODEL_DIR / "unet_barcode.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
    evaluate(model, device, train_size=train_size,
             long_side=args.long_side, use_fallback=use_fallback)