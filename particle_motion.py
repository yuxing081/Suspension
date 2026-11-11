"""单颗粒图像测量与原始数据导出；速度、振动、频谱和事件分析在 particle_postprocess 中。"""

import csv
import json
import shutil
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm
from particle_data import match_trigger, read_flow, write_numeric_csv
from particle_overlay import draw_tracking_overlay
from particle_plots import plot_measurements


def first_particle(gray):
    """首帧检测 ROI 内面积最大、且不接触 ROI 边界的暗颗粒。"""
    height, width = gray.shape
    margin = max(4, width // 24)
    roi = gray[margin:height - margin, margin:width - margin]

    # SciPy ndi.label 默认四连通；明确指定 connectivity=4 保持原行为。
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (roi < 0.55 * np.median(roi)).astype(np.uint8), connectivity=4
    )
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    borders = np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1], 0]
    areas[borders] = 0
    if not areas.max():
        raise ValueError("首帧未找到完整的暗颗粒。")

    particle = int(areas.argmax())
    return centroids[particle] + margin, np.sqrt(areas[particle] / np.pi)


def particle_background(gray, center, radius):
    """用首帧周围亮度填去颗粒所在范围，仅供轮廓分割，不用于运动校正。"""
    background = ndi.gaussian_filter(gray.astype(np.float32), 2)
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    distance = np.hypot(xx - center[0], yy - center[1])
    ring = (distance > 1.55 * radius) & (distance < 2.0 * radius)
    background[distance < 1.55 * radius] = np.median(gray[ring])
    return background


def particle_measurement(gray, background, predicted, radius, with_outline=False):
    x, y = np.rint(predicted).astype(int)
    half = int(np.ceil(2.1 * radius))
    left, top = max(0, x-half), max(0, y-half)
    roi = gray[top:min(y+half+1, gray.shape[0]-32), left:x+half+1]
    base = background[top:top+roi.shape[0], left:left+roi.shape[1]]

    # OpenCV 一次返回连通区域标签和面积；四连通与原 SciPy 算法一致。
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (base - roi > 0.40 * base).astype(np.uint8), connectivity=4
    )
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    particle = int(areas.argmax())
    if not 0.45 * np.pi * radius**2 < areas[particle] < 1.8 * np.pi * radius**2:
        return None

    mask = labels == particle
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return None
    # 填洞后才计算质心/面积，不能直接使用 connectedComponentsWithStats 的未填洞质心。
    mask = ndi.binary_fill_holes(mask)
    cy, cx = ndi.center_of_mass(mask)
    outline = None
    if with_outline:
        ey, ex = np.nonzero(mask & ~ndi.binary_erosion(mask))
        outline = np.column_stack((left + ex, top + ey))
    return np.array([left+cx, top+cy]), mask.sum(), outline


def selected_frame_bounds(frame_count, start_frame=None, end_frame=None):
    """返回视频内从 1 开始、包含两端的有效范围；None 使用对应的整段边界。"""
    first = 1 if start_frame is None else start_frame
    last = frame_count if end_frame is None else end_frame
    if (not isinstance(first, (int, np.integer)) or not isinstance(last, (int, np.integer))
            or not 1 <= first <= last <= frame_count):
        raise ValueError(f"起止帧必须为整数，且满足 1 <= start_frame <= end_frame <= {frame_count}。")
    return first, last


def track_video(video_path, metadata, frame_count, start_frame=None, end_frame=None,
                annotated_video_path=None, annotated_video_fps=None, source_first_frame=1,
                contours_output_path=None):
    """逐帧读取所选视频区间，显示进度，并追踪单颗粒的质心与投影面积。"""
    first, last = selected_frame_bounds(frame_count, start_frame, end_frame)
    selected_count = last - first + 1
    height, width = metadata["height"], metadata["width"]
    position = np.full((selected_count, 2), np.nan)
    area = np.full(selected_count, np.nan)
    predicted = None
    first_gray = None
    first_outline = np.empty((0, 2))
    contour_batches = []
    contour_offsets = np.zeros(selected_count + 1, dtype=np.int64) if contours_output_path is not None else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    writer = None
    try:
        if annotated_video_path is not None:
            annotated_video_path = Path(annotated_video_path)
            if annotated_video_path.resolve() == Path(video_path).resolve():
                raise ValueError("标注视频不能覆盖输入视频。")
            playback_fps = cap.get(cv2.CAP_PROP_FPS) if annotated_video_fps is None else annotated_video_fps
            if not np.isfinite(playback_fps) or playback_fps <= 0:
                raise ValueError("无法确定播放帧率，请设置大于 0 的 annotated_video_fps。")
            annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(playback_fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建标注视频：{annotated_video_path}")
        # 顺序跳过前置帧，保持帧序号准确，不依赖压缩视频的随机定位。
        for _ in range(first - 1):
            if not cap.grab():
                raise ValueError("视频实际帧数少于所选起始帧，或中途解码失败。")
        for i in tqdm(range(selected_count), desc="Tracking", unit="frame", dynamic_ncols=True):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("视频实际帧数少于文件名给出的范围，或中途解码失败。")

            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape != (height, width):
                raise ValueError(f"视频分辨率 {gray.shape[::-1]} 与元数据 {(width, height)} 不一致。")

            if i == 0:
                first_gray = gray.copy()
                predicted, radius = first_particle(gray)
                background = particle_background(gray, predicted, radius)

            outline = None
            if predicted is not None:
                measured = particle_measurement(
                    gray, background, predicted, radius, with_outline=(i == 0 or writer is not None or contours_output_path is not None)
                )
                if measured is None:
                    predicted = None
                else:
                    center, area[i], outline = measured
                    if i == 0:
                        first_outline = outline
                    position[i] = center
                    predicted = center if i == 0 else 2 * center - position[i - 1]

            if contour_offsets is not None:
                if outline is not None:
                    points = outline.astype(np.int32)
                    contour_batches.append(points)
                    contour_offsets[i + 1] = contour_offsets[i] + len(points)
                else:
                    contour_offsets[i + 1] = contour_offsets[i]

            if writer is not None:
                video_frame = first + i
                source_frame = source_first_frame + video_frame - 1
                marked = draw_tracking_overlay(
                    frame=frame, center=position[i], outline=outline,
                    video_frame=video_frame, source_frame=source_frame,
                    time_from_trigger_s=(source_frame - metadata["key_frame_no"]) / metadata["frame_rate"])
                writer.write(marked)

        # 处理到视频尾时，实际尝试读取下一帧，保留原有的总帧数核对。
        if last == frame_count and cap.read()[0]:
            raise ValueError("视频实际帧数多于文件名给出的范围。")
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    if contours_output_path is not None:
        contours_output_path = Path(contours_output_path)
        contours_output_path.parent.mkdir(parents=True, exist_ok=True)
        points = np.concatenate(contour_batches) if contour_batches else np.empty((0, 2), dtype=np.int32)
        source_frames = np.arange(source_first_frame + first - 1, source_first_frame + last, dtype=np.int64)
        np.savez_compressed(contours_output_path, source_frame=source_frames,
                            offsets=contour_offsets, points_px=points)
    return position, area, radius, first_gray, first_outline


def analyze_video(video_path, flow_path, calibration_path, log_path, output_dir=None,
                  start_frame=None, end_frame=None, annotated_video_fps=None):
    """只导出一次视频测量，保留时间/流量同步；派生运动分析由离线入口完成。"""
    video_path = Path(video_path)
    number, first, last = video_path.stem.rsplit("-", 2)
    first, last = int(first), int(last)
    video_frame_count = last - first + 1
    start_frame, end_frame = selected_frame_bounds(video_frame_count, start_frame, end_frame)
    output_dir = video_path.parent / number / "measurements" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(video_path.with_suffix(".json").read_text(encoding="utf-8-sig"))
    frames = np.arange(first + start_frame - 1, first + end_frame)
    fps, key = metadata["frame_rate"], metadata["key_frame_no"]
    # 始终保留相机原始帧号的时间基准，选段首帧不重置为零。
    time = (frames - key) / fps
    calibration_path = Path(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    scale = float(calibration["scale_um_per_px"])
    if calibration["method"] != "axis_aligned_gradient_projection_without_cornerSubPix" or not np.isfinite(scale) or scale <= 0:
        raise ValueError("请先用 run_calibration.py 生成有效的棋盘格标定 JSON。")

    dates, setpoint, raw = read_flow(flow_path)
    trigger = match_trigger(log_path, metadata, first, dates[0], dates[-1])
    sample_time = np.array([(date - trigger).total_seconds() for date in dates])
    if np.any(np.diff(sample_time) <= 0):
        raise ValueError("MFC 时间必须严格递增，才能逐帧插值。")
    raw_frame = np.interp(time, sample_time, raw, left=np.nan, right=np.nan)
    previous = np.searchsorted(sample_time, time, side="right") - 1
    following = np.searchsorted(sample_time, time, side="left")

    annotated_video_path = output_dir / "tracking_overlay.mp4"
    positions, area_px, radius, first_gray, outline = track_video(
        video_path, metadata, video_frame_count, start_frame=start_frame, end_frame=end_frame,
        annotated_video_path=annotated_video_path, annotated_video_fps=annotated_video_fps,
        source_first_frame=first, contours_output_path=output_dir / "contours.npz")
    centroid = positions * scale
    area = area_px * scale**2
    valid = np.isfinite(centroid).all(axis=1)
    if not valid[0]:
        raise ValueError("首帧轮廓分割失败，请检查颗粒与背景的对比。")
    geometry = np.column_stack((positions, area_px, centroid, area))
    with (output_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_frame", "video_frame", "time_from_trigger_s", "pc_time",
                         "centroid_x_px", "centroid_y_px", "projected_area_px2",
                         "centroid_x_um", "centroid_y_um", "projected_area_um2",
                         "tracking_valid", "readflow_raw_interp", "mfc_prev_time_s",
                         "mfc_prev_readflow_raw", "mfc_next_time_s", "mfc_next_readflow_raw"])
        for i, row in enumerate(geometry):
            stamp = trigger + timedelta(seconds=float(time[i]))
            lo, hi = previous[i], following[i]
            samples = (sample_time[lo], raw[lo], sample_time[hi], raw[hi]) if np.isfinite(raw_frame[i]) else (np.nan,) * 4
            values = ["" if not np.isfinite(value) else repr(float(value)) for value in row]
            flow_values = ["" if not np.isfinite(value) else repr(float(value))
                           for value in (raw_frame[i], *samples)]
            writer.writerow([int(frames[i]), start_frame + i, repr(float(time[i])),
                             stamp.isoformat(" ", timespec="microseconds"),
                             *values, int(valid[i]), *flow_values])
    write_numeric_csv(output_dir / "flow_samples.csv",
                      ["time_from_trigger_s", "setpoint_raw", "readflow_raw"],
                      np.column_stack((sample_time, setpoint, raw)), decimal_places=None)
    info = {
        "format_version": 1, "video_path": str(video_path.resolve()),
        "video_filename": video_path.name, "particle_number": int(number),
        "camera_metadata": metadata, "capture_fps": fps, "key_frame_no": key,
        "frame_count": len(frames), "video_frame_count": video_frame_count,
        "start_video_frame": start_frame, "end_video_frame": end_frame,
        "first_source_frame": int(frames[0]), "last_source_frame": int(frames[-1]),
        "source_video_first_frame": first, "source_video_last_frame": last,
        "scale_um_per_px": scale, "initial_radius_px": float(radius),
        "trigger_time": trigger.isoformat(" ", timespec="microseconds"),
        "valid_tracking_frames": int(valid.sum()),
        "calibration_path": str(calibration_path.resolve()),
        "flow_path": str(Path(flow_path).resolve()), "log_path": str(Path(log_path).resolve()),
        "annotated_video_path": str(annotated_video_path.resolve()),
        "centroid_definition": "geometric centroid of thresholded filled particle silhouette",
        "image_axes": "x rightward, y downward; origin is video ROI top-left",
        "contour_definition": "unordered boundary pixels in contours.npz; offsets select each frame",
        "numeric_precision": "float64 round-trip representation; blanks represent missing measurements",
        "flow_reading_unit": "percent of MFC full scale; conversion belongs to postprocessing",
    }
    (output_dir / "measurement_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("calibration.png", "grid_lines_preview.png"):
        check_image = calibration_path.with_name(name)
        if check_image.is_file():
            shutil.copy2(check_image, output_dir / name)
    plot_measurements(output_dir / "measurements.png", time, centroid, area, first_gray, outline)
    print(f"{video_path.stem} | {len(frames)} 帧，{valid.sum()} 帧有效 | 测量输出 {output_dir}")
    return info
