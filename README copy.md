# 砂轮表面 EAN-13 条码识别

本项目对应计算机视觉实验：在砂轮（磨具）表面定位并识别白色条码。
提供 **两条独立技术路线**、**单样本数据增广**以及**统一对比测试程序**。

- 条码真值：`4014855012527`（EAN-13）
- 输入示例：
  - `data/raw/RegulaImg2.bmp`：3467×3467 原始砂轮图，条码位于左下方、带约 20° 倾角；
  - `data/raw/BarAreaRoted.bmp`：528×276 已裁剪条码区域，低分辨率、亮条码/暗背景。

## 1. 目录结构

```text
barcode_wheel/
├── data/
│   ├── raw/                    # 原始两张图（另附一张同型号砂轮图作背景）
│   ├── synthetic/              # 增广后的训练/验证场景 + mask
│   └── test/                   # 16 个真实/退化测试样本 + manifest.json
├── src/barcode_recog/
│   ├── preprocess.py           # CLAHE、锐化、二值化、倾角估计、透视矫正
│   ├── augment.py              # 单样本几何/光度增广、合成场景、测试集增广
│   ├── dataset.py              # PyTorch Dataset
│   ├── traditional.py          # 方法一：梯度张量定位 + 自实现 EAN-13 解码
│   ├── deep_model.py           # 方法二：U-Net 分割 + 滑窗推理 + 解码
│   ├── decode.py               # 95 模块 EAN-13 扫描线解码器（含校验位）
│   └── config.py               # 路径与超参数
├── scripts/
│   ├── prepare_data.py         # 生成增广数据和测试集
│   ├── train_deep.py           # 训练 U-Net
│   ├── benchmark.py            # 两种方法对比测试 + 图表 + 报告
│   └── recognize.py            # 单张图片推理 CLI
├── models/                     # unet_barcode.pt、训练曲线数据
├── outputs/                    # 对比结果、可视化、report.md
├── requirements.txt
└── README.md
```

## 2. 环境与运行

建议 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) 生成增广训练集 + 测试集
python scripts/prepare_data.py

# 2) 训练 U-Net（macOS 自动使用 MPS，否则 CPU）
python scripts/train_deep.py --epochs 30 --size 224 --batch 12

# 3) 对比测试，生成 outputs/report.md、comparison.csv 和图表
python scripts/benchmark.py

# 4) 单张图片识别
python scripts/recognize.py --image data/raw/RegulaImg2.bmp --method both
```

> 本机实验环境中，DSH 自带的 Python 带有 hardened runtime，第三方
> `.so` 会被库校验拦截；因此仓库内的 `run.sh` 使用 ad-hoc 签名的
> Python 启动器。正常安装的 Python 不需要该启动器。

## 3. 方法一：传统计算机视觉

1. **预处理**：缩放长边到 1280，轻度高斯去噪；可选 CLAHE 抑制砂轮反光。
2. **条码候选定位**：对灰度图求 Sobel 梯度，构造结构张量
   `E = Gx²+Gy²` 与方向一致性
   `C = sqrt((Exx-Eyy)² + 4Exy²)/(Exx+Eyy)`。
   条码区域具有“边缘密集且方向高度一致”的特点；对 `E` 取 94.5–99.2 分位数、
   `C>0.45` 做阈值，再闭运算/开运算，连通域即候选条码框。
3. **几何矫正**：对候选区域用 Hough 近垂直线估计条码倾角，旋转矫正；
   再尝试 0°、估计角、反号角及 ±6°/±12° 多角度。
4. **解码**：自实现 EAN-13：
   - 在条码高度 12%–88% 范围内扫描 19 条水平线；
   - 每条线 Otsu 二值化，取最长连续条码段，搜索起止边界；
   - 按 95 个模块（3+6×7+5+6×7+3）采样，匹配 L/G/R 码表与首位奇偶模式；
   - 校验位验证，多扫描线投票。
5. **兜底**：CLAHE/反相/强锐化/Wiener 反卷积变体；
   若整幅图本身就是条码特写，则直接对整图解码。

传统方法不使用训练数据，也不依赖条码 SDK。

## 4. 方法二：深度学习

### 4.1 单样本增广

只有一张真实条码，因此采用 one-shot 合成：

- 从 `BarAreaRoted.bmp` 提取条码标签和像素级条码 mask；
- 从原始砂轮图/同型号砂轮图随机裁背景（排除真实条码区域）；
- 条码随机缩放（占画面 28%–72%）、旋转（±32°）、透视、随机位置；
- 随机亮度、对比度、Gamma、光照梯度、高斯模糊、运动噪声、划痕；
- 生成 260 张训练场景 + 60 张验证场景；另加入约 9% 不含条码的负样本，
  训练时做 hard-negative 重复采样。

### 4.2 网络与训练

- 小型 U-Net：4 次下采样，base channel 16，约 **1.94 M** 参数；
- 损失：`BCEWithLogits(pos_weight=2) + DiceLoss`；
- AdamW + CosineAnnealing，输入 224×224，batch 12，30 epochs；
- 验证集像素 mask IoU 选择最优模型。

### 4.3 测试时推理

- 512×512 重叠滑窗（stride 384），避免整图缩放导致条码过小；
- sigmoid 概率图阈值 0.45，连通域选最大响应；
- `minAreaRect` 得到旋转框，透视矫正后交给同一个 EAN-13 解码器；
- 解码失败时依次尝试轴对齐扩框和整图特写。

## 5. 测试协议

`scripts/benchmark.py` 在 16 个样本上对比：

- 2 个真实样本：原图、给定条码区域图；
- 4 个旋转（±7°、±12°）；
- 1 个透视；
- 5 个光度退化：低照度、过曝、失焦、运动模糊、高斯噪声；
- 低分辨率（55%）、旋转+变暗+模糊组合、JPEG 35、局部遮挡。

指标：

- **定位成功率**：预测框与真值框 IoU ≥ 0.5；
- **端到端识别率**：输出 13 位条码与真值完全一致；
- **CER**：字符错误率（Levenshtein / 13）；
- **耗时**：单张图端到端毫秒数（同一台机器）。

## 6. 输出

运行 `python scripts/benchmark.py` 后：

```text
outputs/
├── comparison.csv              # 逐样本指标
├── results.json                # 汇总 + 明细
├── report.md                   # 自动生成的实验报告
├── figures/
│   ├── accuracy_by_method.png
│   ├── latency.png
│   ├── by_category.png
│   ├── training_curve.png
│   ├── augmentation_examples.jpg
│   ├── test_cases.jpg
│   └── comparison_full.jpg
├── traditional/cases/          # 传统方法逐样本可视化
└── deep/cases/                 # 深度方法逐样本可视化
```

## 7. 说明与局限

- 深度方法是在**同一物理条码**的增广样本上训练，属于 one-shot 实验设置；
  它验证的是“定位+解码流程”，不能等同于跨条码型号的泛化能力。
- 失焦/运动模糊会直接破坏窄条，测试中仍可能失败；工业现场应从
  曝光时间、景深、分辨率（建议条码模块 ≥ 3 px）入手，而不是只靠算法补救。
- 细粒度条码建议在定位后使用原始分辨率裁剪，而不是整图缩放后解码。
