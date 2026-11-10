# 砂轮表面 EAN-13 条码识别

本项目对应计算机视觉实验：在砂轮（磨具）表面定位并识别白色条码。
仓库内提供 **两条独立技术路线**：

| 路线 | 入口 | 核心技术 |
| --- | --- | --- |
| 方法一：传统模式识别 | `barcode_recognize.py` | 结构张量定位 + Hough 倾角矫正 + 自实现 EAN-13 扫描线解码 |
| 方法二：深度学习 | `deep_model_UNet.py`（→ `Unet_model/` 包） | U-Net 语义分割 + 滑窗推理 + 透视矫正 + EAN-13 解码 |

- 条码真值：`4014855012527`（EAN-13，校验位已通过）
- 输入示例：
  - `data/raw/RegulaImg.bmp`：3467×3467 原始砂轮图，条码位于左下方、带约 20° 倾角；
  - `data/raw/BarAreaRoted.bmp`：528×276 已裁剪条码区域，低分辨率、亮条码 / 暗背景；
  - 另有 `BQ-1(1~4)`、`NL-4(2)` 等标签折叠 / 边缘突出的真实样本。

## 1. 目录结构

```text
Grind_Wheel_Barcode_Recognition/
├── barcode_recognize.py        # 方法一：传统方法完整实现 + 命令行入口
├── deep_model_UNet.py          # 方法二：命令行入口（薄壳，转发到 Unet_model 包）
├── Unet_model/                 # 方法二：深度学习包
│   ├── __init__.py             # 包说明，延迟导入 main
│   ├── __main__.py             # 支持 python -m Unet_model
│   ├── config.py               # 路径、全局常量、imread_unicode
│   ├── model.py                # ConvBlock / UNet
│   ├── losses.py               # DiceLoss / CombinedLoss
│   ├── dataset.py              # BarcodeDataset + 真实域数据集生成
│   ├── train.py                # get_device、训练/验证循环、train
│   ├── decode.py               # EAN-13 解码器（扫描线 + zxing 兜底）
│   ├── inference.py            # 滑窗推理、候选框、透视矫正、端到端推理
│   ├── evaluate.py             # 评估与可视化
│   └── main.py                 # 命令行主程序
├── config.py                   # 旧 src 布局遗留（当前未被任何代码引用）
├── utils.py                    # 旧 src 布局遗留（当前未被任何代码引用）
├── data/
│   ├── raw/                    # 原始真实图像（.bmp）
│   ├── real_synth/             # 真实域数据集：train 540 张 / val 66 张 + manifest
│   └── synthetic/              # 旧合成数据集：train 500 张 / val 100 张 + manifest
├── models/                     # unet_barcode.pt、train_history.json
├── outputs/                    # 传统方法可视化结果 + recognize_results.json
│   └── deep/                   # 深度学习推理可视化输出目录
├── requirements.txt
├── run.sh                      # 随附沙箱启动脚本（普通环境无需使用）
├── LICENSE
└── README.md
```

> `config.py` / `utils.py` 属于早期 `src/barcode_recog/` 目录结构的遗留文件，
> 拆分重构后两条流水线均不再导入它们，可忽略。

## 2. 环境与运行

建议 Python 3.10–3.12：

```bash
pip install -r requirements.txt
```

依赖（见 `requirements.txt`）：`numpy>=1.26`、`opencv-python-headless>=4.9`、
`matplotlib>=3.8`、`torch>=2.2`、`zxing-cpp>=2.2`（可选兜底解码器）。

> `run.sh` 是仓库随附的 macOS 沙箱启动器（ad-hoc 签名 Python），
> 在正常安装的 Python 环境下直接调用脚本即可，无需它。

## 3. 方法一：传统计算机视觉

入口 `barcode_recognize.py`，不依赖训练数据，也不依赖条码 SDK。

1. **预处理**：长边缩放到不超过 1280，高斯去噪，Scharr 计算 `gx/gy`。
2. **条码候选定位（结构张量法）**：对 `gx²`、`gy²`、`gx·gy` 做高斯平滑得到
   `Exx/Eyy/Exy`，构造
   - 边缘能量 `E = sqrt(Exx + Eyy)`；
   - 方向一致性 `C = sqrt((Exx-Eyy)² + 4Exy²) / (Exx+Eyy)`。

   条码区域“边缘密集且方向高度一致”，因此在 `E` 的 94 / 97 / 99 分位点阈值化并
   要求 `C ≥ 0.40`；再做闭运算 `(25,13)` / 开运算 `(3,3)`，取连通域，
   按最小尺寸、长宽比 `1~14`、填充率 `≥0.12` 过滤，最后按
   `1.4·填充率 + 1.0·一致性 + 0.25·长宽比` 打分并做 IoU 0.35 的 NMS，保留最多 8 个候选。
3. **几何矫正**：`estimate_bar_angle` 用 Canny + `HoughLinesP` 取近垂直线段，
   以倾角中位数估计条码倾角，`rotate_keep_size` 旋转矫正。
4. **EAN-13 解码（自实现）**：
   - 先对整块求列方向灰度投影（`profile`）做 Otsu 二值化 + 95 模块采样；
   - 再在条码高度 10%–90% 范围扫描 15 条水平线，Otsu 二值化后取最长条码段，
     搜索起止边界，按 95 模块（3 + 6×7 + 5 + 6×7 + 3）采样；
   - 匹配 L/G/R 码表与首位奇偶模式，校验位验证，多扫描线投票取置信度最高者。
5. **兜底链**：每个候选依次尝试
   `zxing → 自实现解码 → 反相 / CLAHE+锐化 → 多角度旋转（±6°/±12°）`；
   若整幅图本身就是条码特写（长宽比 0.8–6.0），再对整图直接解码一次。

输出每个样本的条码值、置信度、校验位、候选数与定位框，并写出可视化结果。

## 4. 方法二：深度学习

### 4.1 数据集

`python deep_model_UNet.py --build-data` 会从 `data/raw` 构建真实域定位数据集：

- 用方法一 `recognize_barcode` 在每张原始图上提取真值框（定位 + 解码验证），
  定位失败的源图会被跳过；
- 每张源图生成若干正 / 负样本（`--data-pos` 默认 60、`--data-neg` 默认 18），
  其中一张源图作为验证集留出（`val_index=1`）；
- 正样本：以真值框为中心随机取场景块（边长约为条码的 1.7–3.0 倍）；
  负样本：随机取远离条码的背景块；
- 场景块经复制填充、缩放后，施加随机旋转（±20°）、透视抖动、光照梯度、
  亮度 / 对比度扰动、高斯模糊与噪声；
- **mask 由真值框经同一变换矩阵映射后 `fillPoly` 得到**，因此是像素级紧贴的精确 mask；
- 输出到 `data/real_synth/{train,val}/images|masks` 与对应 `*_manifest.json`。

仓库内已生成：`real_synth` 训练集 540 张、验证集 66 张；
`data/synthetic`（旧合成数据集，train 500 / val 100）在真实域数据缺失时作为回退。

`BarcodeDataset` 默认一次性预解码并缓存全部样本（避免每轮重复解码），
训练集额外做在线几何 / 光度增广，并按正负样本比例做平衡采样。

### 4.2 网络与训练

- 小型 U-Net：4 次下采样 / 上采样，`base_ch=24`，约 **4.4 M** 参数；
- 损失：`BCEWithLogitsLoss(pos_weight=4) + DiceLoss`；
- 优化器 AdamW（`lr=2e-4`，`weight_decay=1e-5`），线性 warmup（`epochs/15`，至少 8 轮）
  + 余弦退火到 10% 学习率；
- 输入 384×384 灰度图，归一化到 `[-1, 1]`；默认 `batch=16`、`epochs=200`、`dropout=0.1`；
- 以验证集**正样本 mask IoU** 选择最优模型，`--patience 40` 早停；
- 检查点保存 `epoch / model_state / optimizer_state / val_iou / args / train_size`。

`models/` 中当前检查点：`epoch 171`，`val_iou≈0.898`，`train_size=384`
（训练配置为 300 epochs，于第 211 轮早停）。

### 4.3 推理流程

1. 原图长边缩放到 `--long-side`（默认 1280）做粗定位；
2. **批量滑窗推理**：窗口 384、步长 256，右下复制填充对齐后整批前向，
   概率图按重叠次数平均；
3. `extract_candidates`：阈值 0.40 二值化 → 连通域 → 过滤面积占比
   `0.0004~0.60`、长宽比 `1.1~14`、短边 `≥8` → `minAreaRect` 旋转框，
   按 `面积×平均概率` 排序保留前 3；
4. 候选框映射回**全分辨率原图**后 `perspective_correct` 矫正为水平条码
   （关键点：定位在缩放图上做，解码在原图上做，避免小条码被缩到无法解码）；
5. 解码：自研扫描线解码 → 反色 → zxing 兜底；失败则尝试 90°/180°/270° 旋转；
6. 全部失败且未关闭兜底时，回退到方法一 `recognize_barcode`。

### 4.4 包内模块职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 项目路径、`GT_TEXT`、`imread_unicode`（兼容中文路径） |
| `model.py` | `ConvBlock`、`UNet` 网络结构 |
| `losses.py` | `DiceLoss`、`CombinedLoss` |
| `dataset.py` | `BarcodeDataset`、`build_real_dataset` |
| `train.py` | `get_device`、`train_one_epoch`、`validate`、`train` |
| `decode.py` | `decode_ean13`、`decode_barcode_crop`、`valid_ean13` |
| `inference.py` | `sliding_window_infer`、`extract_candidates`、`perspective_correct`、`infer_single` |
| `evaluate.py` | `evaluate`、`visualize_prediction` |
| `main.py` | 命令行参数解析与分支调度 |

## 5. 命令行用法

### 5.1 传统方法

```bash
# 处理 data/raw 下所有图像，结果写入 outputs/
python barcode_recognize.py

# 单张图像
python barcode_recognize.py --image data/raw/RegulaImg.bmp

# 额外输出梯度能量图 + 方向一致性图调试图
python barcode_recognize.py --image data/raw/RegulaImg.bmp --debug
```

### 5.2 深度学习

```bash
# 1) 构建真实域定位数据集（写入 data/real_synth）
python deep_model_UNet.py --build-data --data-pos 60 --data-neg 18

# 2) 训练（结束后自动在验证集 + data/raw 上评估）
python deep_model_UNet.py --epochs 300 --size 384 --batch 16

# 3) 仅评估已训练模型
python deep_model_UNet.py --eval-only

# 4) 单张图像推理（结果写入 outputs/deep/）
python deep_model_UNet.py --infer data/raw/RegulaImg.bmp

# 等价入口
python -m Unet_model --help
```

常用参数（`--help` 可查看全部）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--epochs` | 200 | 训练轮数 |
| `--batch` | 16 | 批大小 |
| `--lr` / `--wd` | 2e-4 / 1e-5 | 学习率 / 权重衰减 |
| `--size` | 384 | 训练图像尺寸 |
| `--base-ch` | 24 | U-Net 基础通道数 |
| `--pos-weight` | 4.0 | BCE 正样本权重 |
| `--dropout` | 0.1 | Dropout 率 |
| `--patience` | 40 | 早停耐心轮数 |
| `--long-side` | 1280 | 推理定位时的图像长边 |
| `--no-fallback` | — | 关闭传统方法兜底，仅评估纯 U-Net |
| `--eval-only` | — | 仅评估已训练模型 |
| `--infer` | — | 对单张图像推理 |
| `--build-data` | — | 构建数据集后退出 |

设备自动选择：MPS → CUDA → CPU。

## 6. 输出

```text
models/
├── unet_barcode.pt             # 最优 U-Net 检查点
└── train_history.json          # train_loss / val_loss / val_iou 曲线数据

outputs/
├── recognize_results.json      # 传统方法逐图结果（JSON）
├── <stem>_result.jpg           # 传统方法可视化（定位框 + 条码文本）
├── <stem>_debug.jpg            # 传统方法调试图（--debug 时生成）
└── deep/
    └── <stem>_deep_result.jpg  # 深度方法可视化（概率热力图 + 定位框）
```

## 7. 说明与局限

- 深度学习路线是在**同一物理条码**的增广样本上训练，属于 one-shot 实验设置；
  它验证的是“定位 + 解码流程”，不等同于跨条码型号的泛化能力。
- 失焦 / 运动模糊会直接破坏窄条，仍可能解码失败；工业现场应从曝光时间、
  景深、分辨率（建议条码模块 ≥ 3 px）入手，而不是只靠算法补救。
- 细粒度条码建议在定位后使用**原始分辨率裁剪**解码，而不是整图缩放后解码——
  这也是深度推理“缩放图定位 + 原图解码”设计的出发点。
- `outputs/` 与 `models/` 会在首次运行时自动创建；`data/real_synth` 缺失时
  `train` 会自动回退到 `data/synthetic`。