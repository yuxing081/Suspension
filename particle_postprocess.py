"""Recompute particle motion results from an exported measurement table."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from scipy import ndimage as ndi
from scipy import signal

from particle_data import load_flow_samples, load_measurements, write_numeric_csv
from particle_plots import plot_directionality, plot_fast_motion, plot_flow, plot_spectrum, plot_vibration


def motion_metrics(position_um, fps, velocity_window_s=.002, vibration_window_s=.020,
                   *, return_covariance=False):
    """Return velocity/RMS and optionally local detrended covariance in µm²."""
    # 初始化结果数组
    velocity = np.full_like(position_um, np.nan)
    rms = np.full(len(position_um), np.nan)
    covariance = np.full((len(position_um), 2, 2), np.nan) if return_covariance else None
    
    
    valid = np.isfinite(position_um).all(axis=1)
    for values in ndi.find_objects(ndi.label(valid)[0]):     # 将轨迹划分成若干连续有效的片段
        section = values[0]
        pos = position_um[section] - np.mean(position_um[section], axis=0)
        for duration, output in ((velocity_window_s, velocity[section]),
                                 (vibration_window_s, rms[section])):
            half = max(1, round(fps * duration / 2))         # 计算窗口大小
            if len(pos) < 2 * half + 1:                      # 如果连续有效数据不够一个完整窗口就不进行计算
                continue
            dt = np.arange(-half, half + 1) / fps            # 当前帧对应窗口范围
            
            # 以下两个函数内部已经实现了滑动窗口的计算
            mean = ndi.uniform_filter1d(pos, len(dt), axis=0)           # 计算局部平均位置
            slope = ndi.correlate1d(pos, dt / (dt @ dt), axis=0)        # 计算局部速度--最小二乘拟合斜率
            
        
            if output.ndim == 2:
                output[half:-half] = slope[half:-half] / 1000
            else:
                variance = ndi.uniform_filter1d(pos**2, len(dt), axis=0) - mean**2
                residual_variance = variance - slope**2 * np.mean(dt**2)
                residual = residual_variance.sum(axis=1)
                output[half:-half] = np.sqrt(np.maximum(residual[half:-half], 0))
                if return_covariance:
                    cross = (ndi.uniform_filter1d(pos[:, 0] * pos[:, 1], len(dt))
                             - mean[:, 0] * mean[:, 1]
                             - slope[:, 0] * slope[:, 1] * np.mean(dt**2))
                    local_covariance = np.empty((len(pos), 2, 2))
                    local_covariance[:, 0, 0] = residual_variance[:, 0]
                    local_covariance[:, 1, 1] = residual_variance[:, 1]
                    local_covariance[:, 0, 1] = local_covariance[:, 1, 0] = cross
                    # 去除矩运算的舍入级残差，避免纯直线漂移产生虚假的优势轴。
                    magnitude = np.max(np.abs(pos), axis=0)
                    roundoff = 64 * np.finfo(float).eps * np.outer(magnitude, magnitude)
                    local_covariance[np.abs(local_covariance) <= roundoff] = 0.
                    linear_axes = np.all(np.abs(np.diff(pos, n=2, axis=0))
                                         <= 64 * np.finfo(float).eps * magnitude, axis=0)
                    local_covariance[:, linear_axes, :] = 0.
                    local_covariance[:, :, linear_axes] = 0.
                    covariance[section][half:-half] = local_covariance[half:-half]
    if return_covariance:
        return velocity, rms, covariance
    return velocity, rms


def directionality_metrics(covariance_um2, time_s, valid, source_frames, video_frames,
                           fps, vibration_window_s, event_index=None):
    """Average complete pre-event residual windows and project RMS by angle."""
    half = max(1, round(fps * vibration_window_s / 2))
    time_s = np.asarray(time_s)
    covariance_um2 = np.asarray(covariance_um2)
    source_frames, video_frames = np.asarray(source_frames), np.asarray(video_frames)
    indices = np.flatnonzero(np.asarray(valid) & np.isfinite(time_s))
    breaks = np.flatnonzero(
        (np.diff(indices) != 1)
        | (np.diff(source_frames[indices]) != 1)
        | (np.diff(video_frames[indices]) != 1)
        | ~np.isclose(np.diff(time_s[indices]), 1 / fps, rtol=1e-6, atol=1e-12)) + 1
    centers = np.zeros(len(time_s), dtype=bool)
    for run in np.split(indices, breaks):
        if len(run) >= 2 * half + 1:
            centers[run[half:-half]] = True
    centers &= np.isfinite(covariance_um2).all(axis=(1, 2))
    if event_index is not None:
        centers &= np.arange(len(time_s)) + half < event_index
    retained = np.flatnonzero(centers)
    angle_deg = np.arange(361, dtype=float)
    summary = {
        "status": "no_valid_windows",
        "window_seconds": float(vibration_window_s),
        "window_count": int(len(retained)),
        "sample_time_range_s": None,
        "window_center_time_range_s": None,
        "mean_covariance_um2": None,
        "principal_axis_deg": None,
        "anisotropy": None,
    }
    if not len(retained):
        return angle_deg, np.full(len(angle_deg), np.nan), summary
    summary["sample_time_range_s"] = time_s[[retained[0] - half, retained[-1] + half]].tolist()
    summary["window_center_time_range_s"] = time_s[retained[[0, -1]]].tolist()
    mean_covariance = covariance_um2[centers].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh((mean_covariance + mean_covariance.T) / 2)
    # 理论上半正定；将浮点误差导致的负特征值截为零。
    eigenvalues = np.maximum(eigenvalues, 0.)
    mean_covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    summary["mean_covariance_um2"] = mean_covariance.tolist()
    theta = np.deg2rad(angle_deg)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    variance = np.einsum("ij,jk,ik->i", directions, mean_covariance, directions)
    directional_rms = np.sqrt(np.maximum(variance, 0.))
    total = float(eigenvalues.sum())
    if total == 0:
        summary["status"] = "zero_power"
    else:
        anisotropy = float((eigenvalues[1] - eigenvalues[0]) / total)
        summary["anisotropy"] = anisotropy
        summary["status"] = "isotropic" if anisotropy <= 1e-6 else "ok"
        if summary["status"] == "ok":
            axis = eigenvectors[:, 1]
            summary["principal_axis_deg"] = float(np.rad2deg(np.arctan2(axis[1], axis[0])) % 180)
    return angle_deg, directional_rms, summary


def departure_index(position_um, velocity, diameter_um, fps):
    """Find the beginning of the final sustained fast-motion segment."""
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
    forward_speed = velocity @ direction * 1000
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
    """Compute the Hann-windowed spectrum of the longest valid pre-event run."""
    positions = position_um[:stop]
    valid = np.isfinite(positions).all(axis=1)
    runs = ndi.find_objects(ndi.label(valid)[0])
    if not runs:
        empty = np.empty((0, 2))
        return np.array([]), empty, None, None
    section = max((run[0] for run in runs), key=lambda item: item.stop - item.start)
    count = section.stop - section.start
    if count < 4:
        empty = np.empty((0, 2))
        return np.array([]), empty, None, None
    traces = positions[section]
    window = signal.windows.hann(count, sym=False)
    fft = np.fft.rfft(signal.detrend(traces, axis=0) * window[:, None], axis=0)
    amplitude = np.abs(fft) / window.sum()
    amplitude[1:(-1 if count % 2 == 0 else None)] *= 2
    frequency = np.fft.rfftfreq(count, d=1 / fps)
    resultant = np.linalg.norm(amplitude, axis=1)
    peaks = signal.find_peaks(resultant)[0]
    peaks = peaks[(peaks >= 3) & (resultant[peaks] > 1e-9)]
    peak = float(frequency[peaks[resultant[peaks].argmax()]]) if len(peaks) else None
    return frequency, amplitude, peak, section


def fit_ramp(sample_time, setpoint, flow_slm, start_flow_slm=5.):
    """Fit the measured ramp from approximately start_flow_slm to trigger time."""
    steps = np.flatnonzero((np.diff(setpoint) > 0) & (sample_time[1:] <= 0)
                           & (np.abs(flow_slm[:-1] - start_flow_slm) < .25))
    if not len(steps):
        raise ValueError(f"触发前没有找到从约 {start_flow_slm:g} SLM 开始的上升段。")
    start = sample_time[steps[-1]]
    keep = (sample_time >= start) & (sample_time < 0)
    at_trigger = np.interp(0, sample_time, flow_slm)
    slope, intercept = np.polyfit(
        np.r_[sample_time[keep], 0.], np.r_[flow_slm[keep], at_trigger], 1)
    return start, slope, intercept, at_trigger


def write_summary(path, row):
    """Export eight scientific summary fields without presentation-only styling."""
    headers = ["颗粒文件名", "序号", "平均投影面积 (µm²)", "面积等效直径 (µm)",
               "振动频谱主峰 (Hz)", "触发截面平均流速 (m/s)",
               "流量上升斜率 (SLM/s)", "快速段平均移动速度 (mm/s)"]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "颗粒结果"
    sheet.append(headers)
    sheet.append(row[:len(headers)])
    sheet.freeze_panes = "A2"
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.column_dimensions["A"].width = 28
    sheet.column_dimensions["B"].width = 12
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _float_value(row, key):
    value = row[key]
    return np.nan if value == "" else float(value)


def _format_number(value, decimal_places=6):
    return "" if not np.isfinite(value) else f"{value:.{decimal_places}f}"


def _load_frame_arrays(rows):
    source_frames = np.array([int(row["source_frame"]) for row in rows], dtype=int)
    time = np.array([float(row["time_from_trigger_s"]) for row in rows])
    centroid_px = np.array([[_float_value(row, "centroid_x_px"),
                             _float_value(row, "centroid_y_px")] for row in rows])
    area_px = np.array([_float_value(row, "projected_area_px2") for row in rows])
    centroid_um = np.array([[_float_value(row, "centroid_x_um"),
                             _float_value(row, "centroid_y_um")] for row in rows])
    area_um2 = np.array([_float_value(row, "projected_area_um2") for row in rows])
    valid = np.array([int(row["tracking_valid"]) == 1 for row in rows])
    return source_frames, time, centroid_px, area_px, centroid_um, area_um2, valid

def _write_frames_csv(path, rows, source_frames, time, centroid_px, area_px,
                      centroid_um, area_um2, displacement, velocity, speed, rms,
                      raw_frame, flow_frame, event_index, event_label):
    columns = ["source_frame", "time_from_trigger_s", "pc_time", "readflow_raw_interp",
               "flow_slm_interp", "centroid_x_um", "centroid_y_um", "projected_area_um2",
               "centroid_x_px", "centroid_y_px", "projected_area_px2", "dx_um", "dy_um",
               "vx_mm_s", "vy_mm_s", "speed_mm_s", "vibration_rms_um", "event",
               "mfc_prev_time_s", "mfc_prev_readflow_raw", "mfc_next_time_s",
               "mfc_next_readflow_raw"]
    values = np.column_stack((raw_frame, flow_frame, centroid_um, area_um2,
                              centroid_px, area_px, displacement, velocity, speed, rms))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for index, values_row in enumerate(values):
            samples = [rows[index][key] for key in
                       ("mfc_prev_time_s", "mfc_prev_readflow_raw",
                        "mfc_next_time_s", "mfc_next_readflow_raw")]
            writer.writerow([int(source_frames[index]), f"{time[index]:.7f}",
                             rows[index]["pc_time"], *(_format_number(value) for value in values_row),
                             event_label if event_index == index else "", *samples])


def postprocess_measurements(measurement_dir, output_dir, event_label="快速移动",
                             event_frame=None, full_scale_slm=500., channel_width_mm=27.,
                             channel_height_mm=12., velocity_window_s=.002,
                             vibration_window_s=.020, wavelet_frequency_range_hz=None,
                             wavelet_frequency_count=160, start_flow_slm=5.):
    """Generate analysis products from exported measurements without raw inputs."""
    
    # 创建文件夹
    measurement_dir = Path(measurement_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(measurement_dir.resolve()):
        raise ValueError("后处理输出必须位于测量目录之外，避免覆盖原始测量文件。")
    if (not np.isfinite(velocity_window_s) or not np.isfinite(vibration_window_s)
            or min(velocity_window_s, vibration_window_s) <= 0):
        raise ValueError("速度和振动窗口必须为正数秒。")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据读入
    info, rows = load_measurements(measurement_dir)
    sample_time, setpoint, raw_samples = load_flow_samples(measurement_dir)
    if not rows:
        raise ValueError("测量目录没有帧记录。")
    fps = float(info["capture_fps"])
    scale = float(info["scale_um_per_px"])
    source_frames, time, centroid_px, area_px, centroid_um, area_um2, valid = _load_frame_arrays(rows)
    if not np.isfinite(centroid_um[0]).all():
        raise ValueError("测量首帧没有有效颗粒质心。")
    
    # 位移、面积、直径随时间变化计算
    displacement = centroid_um - centroid_um[0]
    velocity, rms, covariance = motion_metrics(
        displacement, fps, velocity_window_s, vibration_window_s, return_covariance=True)
    speed = np.linalg.norm(velocity, axis=1)
    mean_area = float(np.nanmean(area_um2))
    diameter = 2 * np.sqrt(mean_area / np.pi)
    
    # 关键帧前后特别计算
    if event_frame is None:
        event_index = departure_index(displacement, velocity, 2 * float(info["initial_radius_px"]) * scale, fps)
    else:
        event_matches = np.flatnonzero(source_frames == int(event_frame))
        if not len(event_matches):
            raise ValueError("人工观察帧不在测量导出范围内。")
        event_index = int(event_matches[0])
    event_time = None if event_index is None else float(time[event_index])
    if event_index is not None:
        mask_start = max(0, event_index - max(1, round(vibration_window_s * fps / 2)))
        rms[mask_start:] = np.nan

    video_frames = np.array([int(row["video_frame"]) for row in rows])
    angle_deg, directional_rms, directionality = directionality_metrics(
        covariance_um2=covariance, time_s=time, valid=valid,
        source_frames=source_frames, video_frames=video_frames, fps=fps,
        vibration_window_s=vibration_window_s, event_index=event_index)
    write_numeric_csv(output_dir / "directionality.csv", ["angle_deg", "rms_um"],
                      np.column_stack((angle_deg, directional_rms)))
    plot_directionality(output_dir / "directionality.png", angle_deg, directional_rms, directionality)


    # 快速移动段计算
    fast = valid & (np.arange(len(source_frames)) >= (event_index if event_index is not None else len(source_frames)))
    fast_rows = np.column_stack((source_frames[fast], time[fast], centroid_um[fast], velocity[fast], speed[fast]))
    
    # 数据写入、、画图
    write_numeric_csv(output_dir / "fast_motion.csv",
                      ["source_frame", "time_from_trigger_s", "centroid_x_um", "centroid_y_um",
                       "vx_mm_s", "vy_mm_s", "speed_mm_s"], fast_rows)
    plot_fast_motion(output_dir / "fast_motion.png", time[fast], centroid_um[fast], velocity[fast])
    plot_vibration(output_dir / "vibration.png", time, displacement, rms, event_time,
                   window_seconds=vibration_window_s)

    frequency, amplitude, peak, spectrum_section = centroid_spectrum(centroid_um, fps, event_index)
    write_numeric_csv(output_dir / "spectrum.csv",
                      ["frequency_hz", "amplitude_x_um", "amplitude_y_um", "amplitude_norm_um"],
                      np.column_stack((frequency, amplitude, np.linalg.norm(amplitude, axis=1))))
    plot_spectrum(output_dir / "spectrum.png", frequency, amplitude, peak)

    trigger = datetime.fromisoformat(str(info["trigger_time"]))
    if np.any(np.diff(sample_time) <= 0):
        raise ValueError("MFC 时间必须严格递增，才能逐帧插值。")
    flow_slm = raw_samples * full_scale_slm / 100
    fit = fit_ramp(sample_time, setpoint, flow_slm, start_flow_slm=start_flow_slm)
    section_m2 = channel_width_mm * channel_height_mm * 1e-6
    trigger_speed = fit[3] * 1e-3 / (60 * section_m2)
    raw_frame = np.array([_float_value(row, "readflow_raw_interp") for row in rows])
    flow_frame = raw_frame * full_scale_slm / 100
    event_flow = None if event_index is None else float(flow_frame[event_index])
    plot_flow(output_dir / "flow.png", sample_time, flow_slm, fit, trigger, event_time,
              event_flow, event_label, trigger_speed)

    _write_frames_csv(output_dir / "frames.csv", rows, source_frames, time,
                      centroid_px, area_px, centroid_um, area_um2, displacement,
                      velocity, speed, rms, raw_frame, flow_frame, event_index, event_label)

    mean_speed_values = speed[fast & np.isfinite(speed)]
    mean_speed = float(mean_speed_values.mean()) if len(mean_speed_values) else None
    write_summary(output_dir / "particle_summary.xlsx", [
        info.get("video_filename", info.get("video_path", "")),
        int(info["particle_number"]), mean_area, float(diameter), peak,
        trigger_speed, fit[1], mean_speed,
    ])
    spectrum_details = None if spectrum_section is None else {
        "first_source_frame": int(source_frames[spectrum_section.start]),
        "last_source_frame": int(source_frames[spectrum_section.stop - 1]),
        "frame_count": int(spectrum_section.stop - spectrum_section.start),
        "frequency_resolution_hz": float(fps / (spectrum_section.stop - spectrum_section.start)),
    }
    event_number = None if event_index is None else int(source_frames[event_index])
    summary = {
        "measurement_dir": str(measurement_dir.resolve()),
        "measurement_format_version": info.get("format_version"),
        "video_path": info.get("video_path"),
        "calibration_path": info.get("calibration_path"),
        "annotated_video_path": info.get("annotated_video_path"),
        "scale_um_per_px": scale,
        "capture_fps": fps,
        "frame_count": len(source_frames),
        "video_frame_count": int(info["video_frame_count"]),
        "start_video_frame": int(info["start_video_frame"]),
        "end_video_frame": int(info["end_video_frame"]),
        "first_source_frame": int(info["first_source_frame"]),
        "last_source_frame": int(info["last_source_frame"]),
        "valid_tracking_frames": int(valid.sum()),
        "background_correction": False,
        "notch_filter": False,
        "centroid_definition": "geometric centroid of thresholded filled particle silhouette",
        "image_axes": "x rightward, y downward; origin is video ROI top-left",
        "vibration_rms_window_s": vibration_window_s,
        "velocity_window_s": velocity_window_s,
        "directionality": directionality,
        "spectrum": spectrum_details,
        "peak_rule": "strongest local amplitude-norm peak with at least 3 cycles in selected segment",
        "mean_area_um2": mean_area,
        "diameter_um": float(diameter),
        "peak_hz": peak,
        "trigger_time": trigger.isoformat(" ", timespec="milliseconds"),
        "trigger_flow_slm": float(fit[3]),
        "trigger_speed_m_s": float(trigger_speed),
        "full_scale_slm": full_scale_slm,
        "channel_width_mm": channel_width_mm,
        "channel_height_mm": channel_height_mm,
        "slope_slm_s": float(fit[1]),
        "event_frame": event_number,
        "requested_event_frame": event_frame,
        "event_label": event_label,
        "event_time_s": event_time,
        "event_flow_slm": event_flow,
        "mean_fast_speed_mm_s": mean_speed,
    }
    if wavelet_frequency_range_hz is not None:
        from particle_wavelet import export_trajectory_wavelet

        summary["wavelet"] = export_trajectory_wavelet(
            output_dir, centroid_um, valid, video_frames, source_frames, time, fps,
            wavelet_frequency_range_hz, wavelet_frequency_count)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{info['video_filename']} | 后处理 {len(source_frames)} 帧 | 结果 {output_dir}")
    return summary
