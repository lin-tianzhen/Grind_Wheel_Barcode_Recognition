"""深度学习砂轮条形码检测识别 — 训练/推理包.

模块划分:
  config     项目路径、全局常量与通用工具
  model      U-Net 网络结构
  losses     DiceLoss / CombinedLoss
  dataset    条码定位数据集与真实域数据集生成
  train      训练与验证循环
  decode     EAN-13 解码器
  inference  滑窗推理、候选框提取、透视矫正与端到端推理
  evaluate   评估与可视化
  main       命令行主程序
"""
from __future__ import annotations

__all__ = ["main"]


def main():
    """延迟导入命令行主程序, 避免包初始化时引入重型依赖."""
    from .main import main as _main
    return _main()