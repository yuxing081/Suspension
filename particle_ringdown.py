"""逐段拟合衰减振动。"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, least_squares

from particle_data import load_measurements, write_numeric_csv
from particle_plots import plot_ringdown, plot_ringdown_selection


def fit_ringdown(time_s, position_um, frequency_range_hz, tau_range_s, *, initial_frequency_hz=None):
    """拟合 q0 + b*t + A*exp(-t/tau)*cos(2*pi*f*t+phi)，t 从零开始。"""
    # 可分离非线性最小二乘拟合，四个线性参数直接lstsq求解，两个非线性参数利用differential_evolution() 全局搜索，再用 least_squares() 局部优化
    
    duration = time_s[-1]
    u = time_s / duration
    position = position_um - position_um[0]

    def design(parameters):
        cycles, decay = parameters
        phase = 2 * np.pi * cycles * u
        envelope = np.exp(-decay * u)
        return np.column_stack((np.ones(len(u)), u,
                                envelope * np.cos(phase), envelope * np.sin(phase)))

    def residual(parameters):
        matrix = design(parameters)
        coefficients = np.linalg.lstsq(matrix, position, rcond=None)[0]
        return matrix @ coefficients - position

    # 对 f、tau 做全局搜索再局部精修；每一步的偏置、漂移、正余弦系数均线性求解。
    bounds = [(frequency_range_hz[0] * duration, frequency_range_hz[1] * duration),
              (duration / tau_range_s[1], duration / tau_range_s[0])]
    lower, upper = np.array(bounds).T
    if initial_frequency_hz is None:
        search = differential_evolution(
            lambda parameters: np.sum(residual(parameters)**2), bounds,
            seed=0, popsize=20, tol=1e-9, polish=False,
        )
        initial = search.x
    else:
        # 比较候选窗口时，用局部频谱给出初值。正式结果仍做全局搜索。
        initial = np.clip([initial_frequency_hz * duration, 1.], lower, upper)
    
    # 最小二乘局部搜索
    fit = least_squares(residual, initial, bounds=(lower, upper),
                        ftol=1e-11, xtol=1e-11, gtol=1e-11)
    
    # 线性参数计算
    matrix = design(fit.x)
    offset, drift, cosine, sine = np.linalg.lstsq(matrix, position, rcond=None)[0]
    amplitude = np.hypot(cosine, sine)
    baseline = position_um[0] + offset + drift * u
    envelope = amplitude * np.exp(-fit.x[1] * u)
    fitted = matrix @ np.array([offset, drift, cosine, sine]) + position_um[0]
    
    # 误差计算
    errors = position_um - fitted
    variance = np.sum((position - position.mean())**2)
    at_bound = np.any(np.isclose(fit.x, np.array(bounds).T, rtol=1e-4, atol=1e-8))
    
    # 输出结果
    result = {
        "frequency_hz": float(fit.x[0] / duration),
        "tau_s": float(duration / fit.x[1]),
        "amplitude_um": float(amplitude),
        "offset_um": float(position_um[0] + offset),
        "drift_um_s": float(drift / duration),
        "phase_rad": float(np.arctan2(-sine, cosine)),
        "rmse_um": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1 - np.sum(errors**2) / variance) if variance else np.nan,
        "at_search_bound": bool(at_bound),
        "optimizer_success": bool(fit.success),
    }
    return result, fitted, baseline, envelope


def _dominant_frequency(detrended, fps, frequency_range_hz):
    """FFT 只用于估计周期和优化初值，频率结果由时域拟合给出。"""
    nfft = 1 << (8 * len(detrended) - 1).bit_length()
    frequencies = np.fft.rfftfreq(nfft, 1 / fps)
    power = np.abs(np.fft.rfft(detrended, n=nfft, axis=0)) ** 2
    if power.ndim == 2:
        power = power.sum(axis=1)
    allowed = (frequencies >= frequency_range_hz[0]) & (frequencies <= frequency_range_hz[1])
    return float(frequencies[allowed][np.argmax(power[allowed])]) if allowed.any() else None


def _oscillation_quality(values, result):
    """相对于仅拟合线性漂移，衰减振荡解释了多少剩余波动。"""
    residual = signal.detrend(values)
    variance = float(residual @ residual)
    return 1 - len(values) * result["rmse_um"]**2 / variance if variance > np.finfo(float).eps else np.nan


def _is_resolved_decay(result, quality, duration, min_cycles, min_oscillation_r2, min_amplitude_um):
    return (result["optimizer_success"] and not result["at_search_bound"]
            and duration * result["frequency_hz"] >= min_cycles
            and quality >= min_oscillation_r2 and result["amplitude_um"] >= min_amplitude_um
            and duration / result["tau_s"] >= -np.log(.8))


def select_ringdown_window(position_um, valid, fps, frequency_range_hz, tau_range_s,
                           min_cycles=3., min_oscillation_r2=.75, min_amplitude_um=.25):
    """在各连续有效段的主波动附近寻找单模态衰减尾段，返回行索引切片和选段方向。"""
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
    best = None
    best_score = -np.inf
    for first, stop in edges:
        if stop - first < 8 or (stop - first - 1) * frequency_range_hz[1] / fps < min_cycles:
            continue
        traces = position_um[first:stop]
        detrended = signal.detrend(traces, axis=0)
        axis = int(np.argmax(np.sum(detrended**2, axis=0)))
        dominant = detrended[:, axis]
        anchor = int(np.argmax(np.abs(dominant)))
        peak = abs(dominant[anchor])
        if peak < min_amplitude_um:
            continue
        frequency = _dominant_frequency(detrended, fps, frequency_range_hz)
        if frequency is None:
            continue
        period = fps / frequency
        # 冲击和慢位移会使整段 FFT 偏低，再从主峰之后估计一次尾段周期。
        tail = traces[anchor + max(1, round(.25 * period)):anchor + round(8 * period)]
        tail_frequency = (_dominant_frequency(signal.detrend(tail, axis=0), fps, frequency_range_hz)
                          if len(tail) >= 8 else None)
        periods = [period] if tail_frequency is None else [period, fps / tail_frequency]
        # 允许跳过主峰附近的起振，同时限制起点向后移动，避免只挑到很弱的尾部。
        candidates = set()
        for period in periods:
            for offset in np.arange(-.25, 1.51, .25):
                start = max(0, anchor + round(offset * period))
                for extra_cycles in (.5, 1.5, 2.5, 3.5, 5., 9., 13.):
                    end = min(len(traces), start + round((min_cycles + extra_cycles) * period) + 1)
                    candidates.add((start, end))
        for start, end in sorted(candidates):
            if end - start < 8 or (end - start - 1) * frequency_range_hz[1] / fps < min_cycles:
                continue
            time = np.arange(end - start) / fps
            values = traces[start:end, axis]
            oscillation = signal.detrend(values)
            variance = float(oscillation @ oscillation)
            if variance <= np.finfo(float).eps:
                continue
            seed = _dominant_frequency(oscillation, fps, frequency_range_hz)
            if seed is None:
                continue
            result, *_ = fit_ringdown(
                time, values, frequency_range_hz, tau_range_s, initial_frequency_hz=seed)
            cycles = time[-1] * result["frequency_hz"]
            quality = 1 - len(time) * result["rmse_um"]**2 / variance
            if not _is_resolved_decay(result, quality, time[-1], min_cycles, min_oscillation_r2,
                                      max(min_amplitude_um, .15 * peak)):
                continue
            # 包络已接近消失的窗口会把过长的平静尾部带入拟合。
            if time[-1] / result["tau_s"] > -np.log(.08):
                continue
            delay_cycles = max(0, start - anchor) * result["frequency_hz"] / fps
            # 质量相近时优先保留更多周期，减少只挑几帧高拟合度数据的倾向。
            score = quality + .03 * min(cycles, 8) - .045 * delay_cycles
            if score > best_score:
                best = (slice(first + start, first + end), axis)
                best_score = score
    return best


def analyze_vibration_segments(measurement_dir, output_dir, segments,
                               frequency_range_hz=(100., 2500.),
                               tau_range_s=(.0001, .1), *, auto_select_window=True,
                               min_cycles=3., min_oscillation_r2=.75, min_amplitude_um=.25):
    """从各搜索区间选出一段衰减尾段，或按指定区间直接拟合。"""
    info, rows = load_measurements(measurement_dir)
    fps = float(info["capture_fps"])
    if not 0 < frequency_range_hz[0] < frequency_range_hz[1] < fps / 2:
        raise ValueError("频率范围须满足 0 < 下限 < 上限 < 采集帧率 / 2。")
    frames = np.array([int(row["source_frame"]) for row in rows])
    video_frames = np.array([int(row["video_frame"]) for row in rows])
    trigger_time = np.array([float(row["time_from_trigger_s"]) for row in rows])
    position = np.array([[float(row["centroid_x_um"] or "nan"),
                          float(row["centroid_y_um"] or "nan")] for row in rows])
    valid = np.array([row["tracking_valid"] == "1" for row in rows]) & np.isfinite(position).all(axis=1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"measurement_dir": str(Path(measurement_dir).resolve()),
                "capture_fps": fps, "segments": segments,
                "frequency_range_hz": frequency_range_hz, "tau_range_s": tau_range_s,
                "auto_select_window": auto_select_window, "min_cycles": min_cycles,
                "min_oscillation_r2": min_oscillation_r2, "min_amplitude_um": min_amplitude_um,
                "model": "q0 + b*t + A*exp(-t/tau)*cos(2*pi*f*t+phi)"}
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    summary, windows = [], []
    for number, start, end in segments:
        search = np.flatnonzero((video_frames >= start) & (video_frames <= end))
        if len(search) < 8 or not np.array_equal(video_frames[search], np.arange(start, end + 1)):
            raise ValueError(f"第 {number} 段须至少 8 帧，且帧号连续并位于测量导出范围内。")
        segment_dir = output_dir / f"{number:02d}_{start}-{end}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        window = {"event_id": number, "search_start_video_frame": start,
                  "search_end_video_frame": end, "start_video_frame": None,
                  "end_video_frame": None, "selection_axis": "", "selection_status": ""}
        windows.append(window)
        selection_title = f"第 {number} 次振动 · 搜索帧 {start}–{end}"
        search_traces = np.where(valid[search, None], position[search], np.nan)
        if auto_select_window:
            selection = select_ringdown_window(
                search_traces, valid[search], fps, frequency_range_hz, tau_range_s,
                min_cycles, min_oscillation_r2, min_amplitude_um)
            if selection is None:
                window["selection_status"] = "no_reliable_window"
                plot_ringdown_selection(segment_dir / "selection.png", video_frames[search],
                                        search_traces, None, selection_title + " · 无合格拟合区间")
                for name in ("fit.csv", "fit.png"):
                    (segment_dir / name).unlink(missing_ok=True)
                print(f"第 {number} 段 {start}–{end}: 未找到满足条件的衰减尾段。")
                continue
            section, axis = selection
            keep = search[section]
            window["selection_axis"] = ("x", "y")[axis]
            window["selection_status"] = "selected"
        else:
            if not valid[search].all():
                raise ValueError(f"第 {number} 段存在无效或非有限跟踪位置。")
            keep = search
            window["selection_status"] = "manual"
        fit_start, fit_end = int(video_frames[keep[0]]), int(video_frames[keep[-1]])
        window.update(start_video_frame=fit_start, end_video_frame=fit_end)
        plot_ringdown_selection(segment_dir / "selection.png", video_frames[search],
                                search_traces, (fit_start, fit_end), selection_title)
        print(f"第 {number} 段: 搜索 {start}–{end} 帧，拟合 {fit_start}–{fit_end} 帧")
        time = (video_frames[keep] - fit_start) / fps
        traces = position[keep]
        results, fitted, baseline, envelope = [], [], [], []
        for axis, values in zip(("x", "y"), traces.T):
            result, curve, trend, decay = fit_ringdown(
                time, values, frequency_range_hz, tau_range_s)
            quality = _oscillation_quality(values, result)
            result["oscillation_r2"] = float(quality)
            resolved = _is_resolved_decay(result, quality, time[-1], min_cycles,
                                          min_oscillation_r2, min_amplitude_um)
            result["fit_status"] = "ok" if resolved else "review"
            results.append(result)
            fitted.append(curve)
            baseline.append(trend)
            envelope.append(decay)
            summary.append({"event_id": number, "start_video_frame": fit_start, "end_video_frame": fit_end,
                            "start_source_frame": int(frames[keep][0]),
                            "end_source_frame": int(frames[keep][-1]),
                            "start_time_from_trigger_s": float(trigger_time[keep][0]),
                            "end_time_from_trigger_s": float(trigger_time[keep][-1]),
                            "axis": axis, **result,
                            "search_start_video_frame": start, "search_end_video_frame": end})
            note = "，参数触及搜索边界" if result["at_search_bound"] else ""
            if not resolved:
                note += "，建议复核"
            print(f"第 {number} 段 {axis.upper()}: f={result['frequency_hz']:.3f} Hz, "
                  f"tau={result['tau_s'] * 1000:.3f} ms, R2={result['r2']:.4f}{note}")
        fitted, baseline, envelope = np.array(fitted).T, np.array(baseline).T, np.array(envelope).T
        
        # 数据输出
        title = f"{selection_title} · 拟合帧 {fit_start}–{fit_end}"
        plot_ringdown(segment_dir / "fit.png", time, traces, fitted, baseline,
                      envelope, results, title)
        write_numeric_csv(
            segment_dir / "fit.csv",
            ["source_frame", "video_frame", "time_from_trigger_s", "time_from_start_s",
             "centroid_x_um", "centroid_y_um", "fitted_x_um", "fitted_y_um",
             "baseline_x_um", "baseline_y_um", "envelope_x_um", "envelope_y_um",
             "residual_x_um", "residual_y_um"],
            np.column_stack((frames[keep], video_frames[keep], trigger_time[keep], time, traces, fitted,
                             baseline, envelope, traces - fitted)), decimal_places=None,
        )
    fit_fields = (
        "event_id", "start_video_frame", "end_video_frame", "start_source_frame",
        "end_source_frame", "start_time_from_trigger_s", "end_time_from_trigger_s",
        "axis", "frequency_hz", "tau_s", "amplitude_um", "offset_um", "drift_um_s",
        "phase_rad", "rmse_um", "r2", "at_search_bound", "optimizer_success",
        "oscillation_r2", "fit_status", "search_start_video_frame", "search_end_video_frame",
    )
    with (output_dir / "fit_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fit_fields)
        writer.writeheader()
        writer.writerows(summary)
    window_fields = ("event_id", "search_start_video_frame", "search_end_video_frame",
                     "start_video_frame", "end_video_frame", "selection_axis", "selection_status")
    with (output_dir / "fit_windows.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=window_fields)
        writer.writeheader()
        writer.writerows(windows)
    return summary
