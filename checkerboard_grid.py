"""基于网格线梯度投影估计物方像素尺寸（µm/px）。

适用于网格线接近水平/垂直、棋盘外围可裁切但指定内格线可见的图像。
只估计像素比例尺，不校正透视或镜头畸变。
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


def _grid_line_positions(gradient_profile, count, margin):
    """从一维绝对梯度投影中选出 count 条峰值较强的网格线。"""
    profile = np.asarray(gradient_profile, dtype=np.float64).copy()
    if count < 2:
        raise ValueError("每个方向至少需要 2 条内网格线。")
    if len(profile) <= 2 * margin:
        raise ValueError("图像太小，边界留白超过了检测范围。")

    profile[:margin] = 0.0
    profile[-margin:] = 0.0
    min_distance = max(1, int(len(profile) / (count + 1) * 0.55))
    candidates, _ = find_peaks(profile, distance=min_distance)
    candidates = candidates[(candidates >= margin) & (candidates < len(profile) - margin)]
    if len(candidates) < count:
        raise ValueError("检测到的网格线不足；请检查 inner_corners 或图像质量。")

    selected = candidates[np.argsort(profile[candidates])[-count:]]
    if profile[selected].min() < 0.25 * profile[selected].max():
        raise ValueError("选中的网格线响应差异过大，请检查是否检测到了背景纹理。")
    return np.sort(selected)


def calibrate_checkerboard(image_path, square_size_mm, inner_corners, output_dir):
    """测量棋盘格的像素物理尺寸，并输出 JSON 和网格线检查图。

    参数：
        image_path: 标定图像路径
        square_size_mm: 单个方格的真实边长，单位 mm
        inner_corners: (列数, 行数)，是完整可见的*内部网格线/内角点*数
        output_dir: 输出文件夹

    返回：包含平均、水平、垂直比例尺及质量警告的 dict。
    """
    if square_size_mm <= 0:
        raise ValueError("square_size_mm 必须大于 0。")
    columns, rows = map(int, inner_corners)
    if columns < 2 or rows < 2:
        raise ValueError("inner_corners 必须是 (列数, 行数)，且每项至少为 2。")

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取图像：{image_path}")

    smooth = gaussian_filter(gray.astype(np.float32), sigma=max(1.0, min(gray.shape) / 200))
    gy, gx = np.gradient(smooth)
    margin = max(12, round(min(gray.shape) * 0.03))
    x_lines = _grid_line_positions(np.abs(gx).mean(axis=0), columns, margin)
    y_lines = _grid_line_positions(np.abs(gy).mean(axis=1), rows, margin)

    dx = np.diff(x_lines).astype(np.float64)
    dy = np.diff(y_lines).astype(np.float64)
    if min(dx.min(), dy.min()) <= 0:
        raise ValueError("网格线坐标有误，无法计算比例尺。")

    # 横、纵方向的尺度分别计算；综合尺度仅供粗略参考。
    square_um = float(square_size_mm) * 1000.0
    horizontal_pitch = float(dx.mean())
    vertical_pitch = float(dy.mean())
    sx = square_um / horizontal_pitch
    sy = square_um / vertical_pitch

    # 每个相邻方格间距按其在网格中出现次数计入综合均值。
    all_pitch = np.r_[np.repeat(dx, rows), np.repeat(dy, columns)]
    mean_pitch = float(all_pitch.mean())
    scale = square_um / mean_pitch
    horizontal_cv = float(dx.std() / horizontal_pitch) if len(dx) > 1 else None
    vertical_cv = float(dy.std() / vertical_pitch) if len(dy) > 1 else None
    directional_disagreement = abs(sx - sy) / ((sx + sy) / 2.0)

    warnings = []
    if horizontal_cv is None or vertical_cv is None:
        warnings.append("某方向只有 1 个相邻间隔，无法评估该方向的间距一致性。")
    if any(cv is not None and cv > 0.02 for cv in (horizontal_cv, vertical_cv)):
        warnings.append("某方向相邻网格间距离散超过 2%，需检查透视、畸变或误检。")
    if directional_disagreement > 0.02:
        warnings.append("横纵比例尺差异超过 2%，不建议直接使用单一 µm/px。")

    result = {
        "image_path": str(image_path.resolve()),
        "image_size_px": [int(gray.shape[1]), int(gray.shape[0])],
        "square_size_mm": float(square_size_mm),
        "inner_corners_columns_rows": [columns, rows],
        "method": "axis_aligned_gradient_projection_without_cornerSubPix",
        "x_grid_lines_px": x_lines.tolist(),
        "y_grid_lines_px": y_lines.tolist(),
        "horizontal_intervals_px": dx.tolist(),
        "vertical_intervals_px": dy.tolist(),
        "horizontal_mean_pitch_px": horizontal_pitch,
        "vertical_mean_pitch_px": vertical_pitch,
        "mean_pitch_px": mean_pitch,
        "horizontal_scale_um_per_px": sx,
        "vertical_scale_um_per_px": sy,
        "scale_um_per_px": scale,
        "horizontal_interval_cv": horizontal_cv,
        "vertical_interval_cv": vertical_cv,
        "directional_scale_disagreement": directional_disagreement,
        "scale_model": "average scale only; no perspective/lens-distortion correction",
        "warnings": warnings,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x in x_lines:
        cv2.line(preview, (int(x), 0), (int(x), gray.shape[0] - 1), (0, 220, 0), 1)
    for y in y_lines:
        cv2.line(preview, (0, int(y)), (gray.shape[1] - 1, int(y)), (0, 220, 0), 1)
    for y in y_lines:
        for x in x_lines:
            cv2.drawMarker(preview, (int(x), int(y)), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("网格线检查图编码失败。")
    encoded.tofile(str(output_dir / "calibration.png"))

    print(f"水平：{horizontal_pitch:.3f} px/格，{sx:.6f} µm/px")
    print(f"垂直：{vertical_pitch:.3f} px/格，{sy:.6f} µm/px")
    print(f"综合（粗略）：{mean_pitch:.3f} px/格，{scale:.6f} µm/px")
    for warning in warnings:
        print(f"警告：{warning}")
    print(f"结果保存在：{output_dir.resolve()}")
    return result
