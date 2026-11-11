"""隔振后的单颗粒投影质心、面积、振动和 MFC 同步；不扣除背景位移。"""

import csv
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from scipy import ndimage as ndi
from scipy import signal
from tqdm import tqdm
from particle_overlay import draw_tracking_overlay
from particle_plots import plot_centroids, plot_spectrum, plot_fast_motion, plot_flow, plot_vibration


def read_flow(flow_path):
    with open(flow_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    times = [datetime.strptime(row["Time"], "%Y/%m/%d %H:%M:%S:%f") for row in rows]
    setpoint = np.array([float(row["SetPoint"]) for row in rows])
    readflow = np.array([float(row["ReadFlow"]) for row in rows])
    return times, setpoint, readflow


def match_trigger(log_path, metadata, first_frame, start, end):
    pattern = re.compile(r"^\[([^|]+)\|INFO\]\[" + re.escape(metadata["ip"]) + r"\]SoftTrigger OK\.")
    triggers = []
    for line in Path(log_path).read_text(encoding="utf-8-sig").splitlines():
        match = pattern.match(line)
        if match:
            trigger = datetime.fromisoformat(match[1])
            if start <= trigger <= end:
                triggers.append(trigger)
    if not triggers:
        raise ValueError("MFC 记录期间没有找到这台相机的 SoftTrigger OK。")
    if len(triggers) == 1:
        return triggers[0]
    # 仅当一个 MFC 文件里有多次触发时，用相机首帧的整秒区分它们。
    camera_start = datetime.fromtimestamp(metadata["timestamp"], timezone.utc).replace(tzinfo=None)
    estimated = camera_start + timedelta(seconds=(metadata["key_frame_no"] - first_frame)
                                         / metadata["frame_rate"])
    return min(triggers, key=lambda t: abs(t - estimated))


def fit_ramp(sample_time, setpoint, flow_slm):
    """从本次上调前的约 5 SLM 实测点拟合到触发（t=0）。"""
    steps = np.flatnonzero((np.diff(setpoint) > 0) & (sample_time[1:] <= 0)
                           & (np.abs(flow_slm[:-1] - 5) < .25))
    if not len(steps):
        raise ValueError("触发前没有找到从约 5 SLM 开始的上升段。")
    start = sample_time[steps[-1]]
    keep = (sample_time >= start) & (sample_time < 0)
    at_trigger = np.interp(0, sample_time, flow_slm)
    slope, intercept = np.polyfit(np.r_[sample_time[keep], 0.], np.r_[flow_slm[keep], at_trigger], 1)
    return start, slope, intercept, at_trigger


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
                annotated_video_path=None, annotated_video_fps=None, source_first_frame=1):
    """逐帧读取所选视频区间，显示进度，并追踪单颗粒的质心与投影面积。"""
    first, last = selected_frame_bounds(frame_count, start_frame, end_frame)
    selected_count = last - first + 1
    height, width = metadata["height"], metadata["width"]
    position = np.full((selected_count, 2), np.nan)
    area = np.full(selected_count, np.nan)
    predicted = None
    first_gray = None
    first_outline = np.empty((0, 2))

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
                    gray, background, predicted, radius, with_outline=(i == 0 or writer is not None)
                )
                if measured is None:
                    predicted = None
                else:
                    center, area[i], outline = measured
                    if i == 0:
                        first_outline = outline
                    position[i] = center
                    predicted = center if i == 0 else 2 * center - position[i - 1]

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

    return position, area, radius, first_gray, first_outline



def motion_metrics(position_um, fps):
    """速度为 2 ms 局部直线斜率，振动为 20 ms 窗内去线性趋势后的二维 RMS。"""
    velocity = np.full_like(position_um, np.nan)
    rms = np.full(len(position_um), np.nan)
    for values in ndi.find_objects(ndi.label(np.isfinite(position_um).all(axis=1))[0]):
        section = values[0]
        # 平移不改变斜率或去趋势 RMS；先居中，减轻二阶矩相减的舍入误差。
        pos = position_um[section] - np.mean(position_um[section], axis=0)
        for duration, output in ((.002, velocity[section]), (.020, rms[section])):
            half = max(1, round(fps * duration / 2))
            if len(pos) < 2 * half + 1:
                continue
            dt = np.arange(-half, half+1) / fps
            mean = ndi.uniform_filter1d(pos, len(dt), axis=0)
            slope = ndi.correlate1d(pos, dt / (dt @ dt), axis=0)
            if output.ndim == 2:
                output[half:-half] = slope[half:-half] / 1000  # µm/s → mm/s
            else:
                variance = ndi.uniform_filter1d(pos**2, len(dt), axis=0) - mean**2
                residual = (variance - slope**2 * np.mean(dt**2)).sum(axis=1)
                output[half:-half] = np.sqrt(np.maximum(residual[half:-half], 0))
    return velocity, rms


def departure_index(position_um, velocity, diameter_um, fps):
    """在最终离开原停留区的轨迹上，找持续快速移动的起点，排除移出后又返回的振动。"""
    hold = max(1, round(.002 * fps))
    ahead = max(2, round(.015 * fps))
    valid = np.flatnonzero(np.isfinite(position_um).all(axis=1))
    if len(valid) <= hold:
        return None
    origin = np.median(position_um[valid[:hold]], axis=0)
    distance = np.linalg.norm(position_um - origin, axis=1)
    near = np.flatnonzero(distance < diameter_um / 2)
    if not len(near) or distance[valid[-1]] < diameter_um:
        return None
    crossing = near[-1] + 1
    direction = (position_um[crossing] - origin) / distance[crossing]
    forward_speed = velocity @ direction * 1000  # µm/s
    rapid = forward_speed >= 5 * diameter_um
    edges = np.flatnonzero(np.diff(np.r_[False, rapid, False])).reshape(-1, 2)
    for start, end in edges:
        if end <= crossing or start > crossing + ahead or end - start < hold:
            continue
        later = min(start + ahead, valid[-1])
        advance = (position_um[later] - position_um[start]) @ direction
        if advance >= diameter_um / 2:
            return int(start)
    return None


def centroid_spectrum(position_um, fps, stop):
    """取移走前最长的连续有效轨迹；去线性趋势、加 Hann 窗，求单边位移幅度。"""
    positions = position_um[:stop]
    valid = np.isfinite(positions).all(axis=1)
    runs = ndi.find_objects(ndi.label(valid)[0])
    if not runs:
        empty = np.empty((0, 2))
        return np.array([]), empty, None, None
    section = max((run[0] for run in runs), key=lambda s: s.stop - s.start)
    n = section.stop - section.start
    if n < 4:
        empty = np.empty((0, 2))
        return np.array([]), empty, None, None
    traces = positions[section]
    window = signal.windows.hann(n, sym=False)
    fft = np.fft.rfft(signal.detrend(traces, axis=0) * window[:, None], axis=0)
    amplitude = np.abs(fft) / window.sum()
    amplitude[1:(-1 if n % 2 == 0 else None)] *= 2
    frequency = np.fft.rfftfreq(n, d=1/fps)
    resultant = np.linalg.norm(amplitude, axis=1)
    peaks = signal.find_peaks(resultant)[0]
    peaks = peaks[(peaks >= 3) & (resultant[peaks] > 1e-9)]
    # 在窗口内至少有三个周期的局部谱峰中取最强者；谱峰只描述测得的质心波动。
    peak = float(frequency[peaks[resultant[peaks].argmax()]]) if len(peaks) else None
    return frequency, amplitude, peak, section


def write_numeric_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow("" if not np.isfinite(v) else f"{v:.8f}".rstrip("0").rstrip(".")
                            for v in row)


def write_summary(path, row):
    """用 openpyxl 直接写入原有八列颗粒汇总。"""
    headers = ["颗粒文件名", "序号", "平均投影面积 (µm²)", "面积等效直径 (µm)",
               "振动频谱主峰 (Hz)", "触发截面平均流速 (m/s)",
               "流量上升斜率 (SLM/s)", "快速段平均移动速度 (mm/s)"]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "颗粒结果"
    sheet.sheet_view.showGridLines = False
    sheet.append(headers)
    sheet.append(row[:len(headers)])
    edge = Side(style="thin", color="D9E2F3")
    for cells in sheet:
        for cell in cells:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(vertical="center")
            cell.border = Border(left=edge, right=edge, top=edge, bottom=edge)
    for cell in sheet[1]:
        cell.font = Font(name="Microsoft YaHei", size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in sheet[2][2:]:
        cell.number_format = "0.000"
    for column, width in zip("ABCDEFGH", [28, 10, 18, 18, 18, 18, 18, 18]):
        sheet.column_dimensions[column].width = width
    sheet.row_dimensions[1].height = 34
    sheet.freeze_panes = "A2"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def analyze_video(video_path, flow_path, calibration_path, log_path, output_dir=None,
                  event_label="快速移动", event_frame=None, full_scale_slm=500.,
                  channel_width_mm=27., channel_height_mm=12.,
                  start_frame=None, end_frame=None, annotated_video_fps=None):
    """单视频、单颗粒；起止帧为视频内序号，event_frame 为人工原始帧号。"""
    video_path = Path(video_path)
    number, first, last = video_path.stem.rsplit("-", 2)
    first, last = int(first), int(last)
    video_frame_count = last - first + 1
    start_frame, end_frame = selected_frame_bounds(video_frame_count, start_frame, end_frame)
    output_dir = video_path.parent / number if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(video_path.with_suffix(".json").read_text(encoding="utf-8-sig"))
    frames = np.arange(first + start_frame - 1, first + end_frame)
    fps, key = metadata["frame_rate"], metadata["key_frame_no"]
    # 保留相机原始帧号的时间基准，选段首帧不重新计为 t=0。
    time = (frames - key) / fps

    # 1. 读取独立棋盘格标定；ROI 裁剪不改变物方像素尺寸。
    calibration_path = Path(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    scale = float(calibration["scale_um_per_px"])
    if calibration["method"] != "axis_aligned_gradient_projection_without_cornerSubPix" or not np.isfinite(scale) or scale <= 0:
            raise ValueError("请先用 run_calibration.py 生成有效的棋盘格标定 JSON。")

    # 先确认流量和触发信息，避免视频计算完成后才发现输入不匹配。
    dates, setpoint, raw = read_flow(flow_path)
    # 用导出视频的原始首帧匹配触发，不能改成所选区间的首帧。
    trigger = match_trigger(log_path, metadata, first, dates[0], dates[-1])
    sample_time = np.array([(date - trigger).total_seconds() for date in dates])
    if np.any(np.diff(sample_time) <= 0):
        raise ValueError("MFC 时间必须严格递增，才能逐帧插值。")
    flow_slm = raw * full_scale_slm / 100
    fit = fit_ramp(sample_time, setpoint, flow_slm)

    annotated_video_path = output_dir / "tracking_overlay.mp4"
    positions, area_px, radius, first_gray, outline = track_video(
        video_path, metadata, video_frame_count, start_frame=start_frame, end_frame=end_frame,
        annotated_video_path=annotated_video_path, annotated_video_fps=annotated_video_fps,
        source_first_frame=first)
    centroid = positions * scale
    if not np.isfinite(centroid[0]).all():
        raise ValueError("首帧轮廓分割失败，请检查颗粒与背景的对比。")
    area = area_px * scale**2
    displacement = centroid - centroid[0]
    velocity, rms = motion_metrics(displacement, fps)
    speed = np.linalg.norm(velocity, axis=1)
    if event_frame is None:
        index = departure_index(displacement, velocity, 2 * radius * scale, fps)
    else:
        if not frames[0] <= event_frame <= frames[-1]:
            raise ValueError("人工观察帧不在本视频导出范围内。")
        index = int(event_frame - frames[0])
    event_time = None if index is None else float(time[index])
    if index is not None:
        rms[max(0, index - max(1, round(.010 * fps))):] = np.nan

    # 2. 快速移动单独记录，频谱只用快速移动之前的连续有效段。
    fast = np.isfinite(centroid).all(axis=1) & (np.arange(len(frames)) >= (len(frames) if index is None else index))
    fast_rows = np.column_stack((frames[fast], time[fast], centroid[fast], velocity[fast], speed[fast]))
    write_numeric_csv(output_dir / "fast_motion.csv",
                      ["source_frame", "time_from_trigger_s", "centroid_x_um", "centroid_y_um",
                       "vx_mm_s", "vy_mm_s", "speed_mm_s"], fast_rows)
    plot_fast_motion(output_dir / "fast_motion.png", time[fast], centroid[fast], velocity[fast])
    plot_centroids(output_dir / "centroids.png", time, centroid, area, event_time, first_gray, outline)
    plot_vibration(output_dir / "vibration.png", time, displacement, rms, event_time)
    freq, amp, peak, spectrum_section = centroid_spectrum(centroid, fps, index)
    write_numeric_csv(output_dir / "spectrum.csv",
                      ["frequency_hz", "amplitude_x_um", "amplitude_y_um", "amplitude_norm_um"],
                      np.column_stack((freq, amp, np.linalg.norm(amp, axis=1))))
    plot_spectrum(output_dir / "spectrum.png", freq, amp, peak)

    # 3. MFC 百分读数换为 SLM。1 SLM = 1e-3 / 60 m³/s，截面由 mm² 换成 m²。
    section_m2 = channel_width_mm * channel_height_mm * 1e-6
    trigger_speed = fit[3] * 1e-3 / (60 * section_m2)
    raw_frame = np.interp(time, sample_time, raw, left=np.nan, right=np.nan)
    flow_frame = raw_frame * full_scale_slm / 100
    event_flow = None if index is None else float(flow_frame[index])
    plot_flow(output_dir / "flow.png", sample_time, flow_slm, fit, trigger, event_time,
              event_flow, event_label, trigger_speed)

    previous = np.searchsorted(sample_time, time, side="right") - 1
    following = np.searchsorted(sample_time, time, side="left")
    with (output_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_frame", "time_from_trigger_s", "pc_time", "readflow_raw_interp",
                         "flow_slm_interp", "centroid_x_um", "centroid_y_um", "projected_area_um2",
                         "centroid_x_px", "centroid_y_px", "projected_area_px2",
                         "dx_um", "dy_um", "vx_mm_s", "vy_mm_s", "speed_mm_s", "vibration_rms_um",
                         "event", "mfc_prev_time_s", "mfc_prev_readflow_raw", "mfc_next_time_s", "mfc_next_readflow_raw"])
        metrics = np.column_stack((raw_frame, flow_frame, centroid, area, positions, area_px,
                                   displacement, velocity, speed, rms))
        for i, row in enumerate(metrics):
            stamp = trigger + timedelta(seconds=float(time[i]))
            lo, hi = previous[i], following[i]
            samples = (sample_time[lo], raw[lo], sample_time[hi], raw[hi]) if np.isfinite(raw_frame[i]) else ("",)*4
            writer.writerow([int(frames[i]), f"{time[i]:.7f}", stamp.isoformat(" ", timespec="microseconds"),
                             *(f"{v:.6f}" if np.isfinite(v) else "" for v in row),
                             event_label if i == index else "", *samples])

    # 4. 一行记录一个颗粒。直径按平均投影面积折算，速度取快速段有效合速度的均值。
    mean_area = float(np.nanmean(area))
    diameter = float(2 * np.sqrt(mean_area / np.pi))
    fast_speed = speed[fast & np.isfinite(speed)]
    mean_speed = float(fast_speed.mean()) if len(fast_speed) else None
    write_summary(output_dir / "particle_summary.xlsx", [video_path.name, int(number), mean_area,
                  diameter, peak, trigger_speed, fit[1], mean_speed])
    event_number = None if index is None else int(frames[index])
    spectrum_details = None if spectrum_section is None else {
        "first_source_frame": int(frames[spectrum_section.start]),
        "last_source_frame": int(frames[spectrum_section.stop - 1]),
        "frame_count": spectrum_section.stop - spectrum_section.start,
        "frequency_resolution_hz": fps / (spectrum_section.stop - spectrum_section.start),
    }
    summary = {
        "video_path": str(video_path.resolve()),
        "calibration_path": str(calibration_path.resolve()),
        "scale_um_per_px": scale, "capture_fps": fps, "frame_count": len(frames),
        "video_frame_count": video_frame_count,
        "start_video_frame": start_frame, "end_video_frame": end_frame,
        "annotated_video_path": str(annotated_video_path.resolve()),
        "valid_tracking_frames": int(np.isfinite(centroid).all(axis=1).sum()),
        "background_correction": False, "notch_filter": False,
        "centroid_definition": "geometric centroid of thresholded filled particle silhouette",
        "image_axes": "x rightward, y downward; origin is video ROI top-left",
        "vibration_rms_window_s": .020, "velocity_window_s": .002,
        "spectrum": spectrum_details,
        "peak_rule": "strongest local amplitude-norm peak with at least 3 cycles in selected segment",
        "mean_area_um2": mean_area, "diameter_um": diameter, "peak_hz": peak,
        "trigger_time": trigger.isoformat(" ", timespec="milliseconds"),
        "trigger_flow_slm": fit[3], "trigger_speed_m_s": trigger_speed,
        "full_scale_slm": full_scale_slm,
        "channel_width_mm": channel_width_mm, "channel_height_mm": channel_height_mm,
        "slope_slm_s": fit[1], "event_frame": event_number, "event_time_s": event_time,
        "event_flow_slm": event_flow, "mean_fast_speed_mm_s": mean_speed,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    check_image = calibration_path.with_name("calibration.png")
    if check_image.is_file():
        shutil.copy2(check_image, output_dir / "calibration.png")
    print(f"{video_path.stem} | {scale:.4f} µm/px | {trigger} 触发 {fit[3]:.2f} SLM，"
          f"{trigger_speed:.3f} m/s | 斜率 {fit[1]:.3f} SLM/s | 结果 {output_dir}")
    return summary
