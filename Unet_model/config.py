"""项目路径、全局常量与通用工具."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
REAL_DIR = PROJECT_ROOT / "data" / "real_synth"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "deep"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_TEXT = "4014855012527"


def imread_unicode(path, flags: int = cv2.IMREAD_GRAYSCALE):
    """读取图像, 兼容含中文等非 ASCII 字符的 Windows 路径.

    cv2.imread 在 Windows 下无法打开非 ASCII 路径, 改为按字节流读取后解码.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)