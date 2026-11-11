"""逐段拟合有漂移的单模态衰减振动，并计算复 Morlet 连续小波变换。"""

import csv
import json
from pathlib import Path

import numpy as np
import pywt
from scipy import signal
from scipy.optimize import differential_evolution, least_squares

from particle_data import load_measurements, write_numeric_csv
from particle_plots import plot_ringdown, plot_ringdown_wavelet


def fit_ringdown(time_s, position_um, frequency_range_hz, tau_range_s):
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
    # 差分进化全局搜索
    search = differential_evolution(
        lambda parameters: np.sum(residual(parameters)**2), bounds,
        seed=0, popsize=20, tol=1e-9, polish=False,
    )
    
    # 最小二乘局部搜索
    fit = least_squares(residual, search.x, bounds=np.array(bounds).T,
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


def morlet_power(position_um, fps, frequency_range_hz, frequency_count):
    """原始位置仅去线性趋势；返回 |CWT|²/scale，非 PSD 或校准振幅。"""
    wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
    frequency = np.geomspace(*frequency_range_hz, frequency_count)
    scales = pywt.frequency2scale(wavelet, frequency / fps)
    detrended = signal.detrend(position_um, axis=0)
    coefficients, frequency = pywt.cwt(
        detrended.T, scales, wavelet, sampling_period=1 / fps, method="fft",
        precision=12,
    )
    power = np.moveaxis(np.abs(coefficients)**2 / scales[:, None, None], 1, 0)
    # cmor 的高斯包络功率降至 e^-2 的半宽：sqrt(B)*scale/fps，作为边缘影响参考。
    edge_time = np.sqrt(wavelet.bandwidth_frequency) * scales / fps
    return frequency, power, edge_time


def analyze_vibration_segments(measurement_dir, output_dir, segments,
                               frequency_range_hz=(100., 2500.),
                               tau_range_s=(.0001, .1), wavelet_frequency_count=160):
    """按原视频帧号闭区间逐段分析，仅写入指定的后处理目录。"""
    info, rows = load_measurements(measurement_dir)
    fps = float(info["capture_fps"])
    if not 0 < frequency_range_hz[0] < frequency_range_hz[1] < fps / 2:
        raise ValueError("频率范围须满足 0 < 下限 < 上限 < 采集帧率 / 2。")
    frames = np.array([int(row["source_frame"]) for row in rows])
    video_frames = np.array([int(row["video_frame"]) for row in rows])
    trigger_time = np.array([float(row["time_from_trigger_s"]) for row in rows])
    position = np.array([[float(row["centroid_x_um"] or "nan"),
                          float(row["centroid_y_um"] or "nan")] for row in rows])
    valid = np.array([row["tracking_valid"] == "1" for row in rows])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"measurement_dir": str(Path(measurement_dir).resolve()),
                "capture_fps": fps, "segments": segments,
                "frequency_range_hz": frequency_range_hz, "tau_range_s": tau_range_s,
                "wavelet": "cmor1.5-1.0", "wavelet_frequency_count": wavelet_frequency_count,
                "model": "q0 + b*t + A*exp(-t/tau)*cos(2*pi*f*t+phi)"}
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = []
    for number, start, end in segments:
        keep = (video_frames >= start) & (video_frames <= end)
        if (keep.sum() < 8 or not np.array_equal(video_frames[keep], np.arange(start, end + 1))
                or not valid[keep].all() or not np.isfinite(position[keep]).all()):
            raise ValueError(f"第 {number} 段须至少 8 帧，且在测量导出范围内连续有效。")
        time = (video_frames[keep] - start) / fps
        traces = position[keep]
        results, fitted, baseline, envelope = [], [], [], []
        for axis, values in zip(("x", "y"), traces.T):
            result, curve, trend, decay = fit_ringdown(
                time, values, frequency_range_hz, tau_range_s)
            results.append(result)
            fitted.append(curve)
            baseline.append(trend)
            envelope.append(decay)
            summary.append({"event_id": number, "start_video_frame": start, "end_video_frame": end,
                            "start_source_frame": int(frames[keep][0]),
                            "end_source_frame": int(frames[keep][-1]),
                            "start_time_from_trigger_s": float(trigger_time[keep][0]),
                            "end_time_from_trigger_s": float(trigger_time[keep][-1]),
                            "axis": axis, **result})
            note = "，参数触及搜索边界" if result["at_search_bound"] else ""
            print(f"第 {number} 段 {axis.upper()}: f={result['frequency_hz']:.3f} Hz, "
                  f"tau={result['tau_s'] * 1000:.3f} ms, R2={result['r2']:.4f}{note}")
        fitted, baseline, envelope = np.array(fitted).T, np.array(baseline).T, np.array(envelope).T
        
        # 小波变换
        frequency, power, edge_time = morlet_power(
            traces, fps, frequency_range_hz, wavelet_frequency_count)
        
        # 数据输出
        segment_dir = output_dir / f"{number:02d}_{start}-{end}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        title = f"第 {number} 次振动 · 原视频帧 {start}–{end}"
        plot_ringdown(segment_dir / "fit.png", time, traces, fitted, baseline,
                      envelope, results, title)
        plot_ringdown_wavelet(segment_dir / "wavelet.png", time, frequency, power,
                              edge_time, results, title)
        write_numeric_csv(
            segment_dir / "fit.csv",
            ["source_frame", "video_frame", "time_from_trigger_s", "time_from_start_s",
             "centroid_x_um", "centroid_y_um", "fitted_x_um", "fitted_y_um",
             "baseline_x_um", "baseline_y_um", "envelope_x_um", "envelope_y_um",
             "residual_x_um", "residual_y_um"],
            np.column_stack((frames[keep], video_frames[keep], trigger_time[keep], time, traces, fitted,
                             baseline, envelope, traces - fitted)), decimal_places=None,
        )
        np.savez_compressed(segment_dir / "wavelet.npz", source_frame=frames[keep],
                            video_frame=video_frames[keep],
                            time_from_start_s=time, frequency_hz=frequency,
                            normalized_power_x=power[0], normalized_power_y=power[1],
                            edge_time_s=edge_time)
    with (output_dir / "fit_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return summary
