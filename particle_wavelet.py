"""连续有效颗粒轨迹的复 Morlet 小波分析。"""

import re
from pathlib import Path

import numpy as np
import pywt
from scipy import signal

from particle_plots import plot_trajectory_wavelet


def morlet_power(position_um, fps, frequency_range_hz, frequency_count):
    """各方向去线性趋势，返回频率、|CWT|²/scale 及边缘参考宽度。"""
    wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
    frequency = np.geomspace(*frequency_range_hz, frequency_count)
    scales = pywt.frequency2scale(wavelet, frequency / fps)
    detrended = signal.detrend(position_um, axis=0)
    coefficients, frequency = pywt.cwt(
        detrended.T, scales, wavelet, sampling_period=1 / fps, method="fft",
        precision=12,
    )
    power = np.abs(coefficients)
    power **= 2
    power /= scales[:, None, None]
    # cmor 的高斯包络功率降至 e^-2 的半宽。
    edge_time = np.sqrt(wavelet.bandwidth_frequency) * scales / fps
    return frequency, np.moveaxis(power, 1, 0), edge_time


def export_trajectory_wavelet(output_dir, position_um, valid, video_frames,
                              source_frames, trigger_time_s, fps,
                              frequency_range_hz, frequency_count):
    """分段计算小波，按原时间出图，用半开偏移量标记 NPZ 中各段的列。"""
    if not 0 < frequency_range_hz[0] < frequency_range_hz[1] < fps / 2:
        raise ValueError("小波频率范围须满足 0 < 下限 < 上限 < 采集帧率 / 2。")
    if frequency_count < 2:
        raise ValueError("小波频率点数须至少为 2。")

    output_dir = Path(output_dir)
    indices = np.flatnonzero(valid & np.isfinite(position_um).all(axis=1))
    breaks = np.flatnonzero((np.diff(indices) != 1)
                            | (np.diff(video_frames[indices]) != 1)
                            | (np.diff(source_frames[indices]) != 1)) + 1
    runs = [run for run in np.split(indices, breaks) if len(run) >= 2]
    expected_plots = set()
    summary = None
    if runs:
        segment_offsets = np.r_[0, np.cumsum([len(run) for run in runs])]
        retained = np.concatenate(runs)
        power = np.empty((2, frequency_count, len(retained)))
        edge_mask = np.empty((frequency_count, len(retained)), dtype=bool)
        sections = [slice(start, stop) for start, stop in zip(segment_offsets[:-1], segment_offsets[1:])]
        for run, section in zip(runs, sections):
            frequency, power[:, :, section], edge_time = morlet_power(
                position_um[run], fps, frequency_range_hz, frequency_count)
            elapsed = (video_frames[run] - video_frames[run[0]]) / fps
            edge_mask[:, section] = (np.minimum(elapsed, elapsed[-1] - elapsed)
                                     < edge_time[:, None])

        power_max = float(power.max())
        np.savez_compressed(
            output_dir / "wavelet.npz", video_frame=video_frames[retained],
            source_frame=source_frames[retained], time_from_trigger_s=trigger_time_s[retained],
            time_from_start_s=trigger_time_s[retained] - trigger_time_s[0],
            segment_offsets=segment_offsets, frequency_hz=frequency,
            normalized_power_x=power[0], normalized_power_y=power[1],
            edge_time_s=edge_time, edge_mask=edge_mask, power_max=power_max,
        )
        for number, (run, section) in enumerate(zip(runs, sections), start=1):
            name = "wavelet.png" if len(runs) == 1 else f"wavelet_{number:02d}.png"
            first, last = video_frames[run[[0, -1]]]
            plot_trajectory_wavelet(
                output_dir / name, trigger_time_s[run], position_um[run], frequency,
                power[:, :, section], edge_mask[:, section],
                f"连续有效轨迹 · 原视频帧 {first}–{last}", power_max)
            expected_plots.add(name)
        summary = {
            "frequency_range_hz": list(frequency_range_hz),
            "frequency_count": int(frequency_count),
            "segment_count": len(runs),
            "frame_count": len(retained),
        }
    else:
        (output_dir / "wavelet.npz").unlink(missing_ok=True)

    # 有效段数量改变时，清理同目录中旧的小波图。
    for path in output_dir.glob("wavelet*.png"):
        if re.fullmatch(r"wavelet(?:_\d{2,})?\.png", path.name) and path.name not in expected_plots:
            path.unlink()
    return summary
