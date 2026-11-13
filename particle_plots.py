"""Plots for particle motion, vibration and MFC flow."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm, Normalize
import numpy as np


def _style():
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                 if name in names), "DejaVu Sans")
    return {"font.family": font, "axes.unicode_minus": False,
            "axes.spines.top": False, "axes.spines.right": False}


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)




def plot_measurements(output_path, time, centroid_um, area_um2,
                      first_gray=None, outline_xy=None):
    """Save the basic first-frame, centroid and projected-area measurements."""
    time = np.asarray(time)
    centroid_um = np.asarray(centroid_um)
    area_um2 = np.asarray(area_um2)
    with plt.rc_context(_style()):
        fig = plt.figure(figsize=(11, 7), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        image_ax = fig.add_subplot(grid[0, 0])
        centroid_ax = fig.add_subplot(grid[0, 1])
        area_ax = fig.add_subplot(grid[1, :])
        if first_gray is None:
            image_ax.axis("off")
        else:
            image_ax.imshow(first_gray, cmap="gray")
            if outline_xy is not None and len(outline_xy):
                outline_xy = np.asarray(outline_xy)
                image_ax.scatter(outline_xy[:, 0], outline_xy[:, 1], s=2,
                                 facecolors="none", edgecolors="#e45756",
                                 label="颗粒边缘")
                image_ax.legend(loc="upper right", frameon=False)
            image_ax.set(title="首帧颗粒轮廓", xlabel="像素 x", ylabel="像素 y")
        centroid_ax.plot(time, centroid_um[:, 0], label="X", lw=.9)
        centroid_ax.plot(time, centroid_um[:, 1], label="Y", lw=.9)
        centroid_ax.legend(frameon=False)
        centroid_ax.set(title="质心随时间", xlabel="相对触发时间（s）", ylabel="质心（µm）")
        area_ax.plot(time, area_um2, color="#28786d", lw=.9)
        area_ax.set(title="颗粒投影面积随时间", xlabel="相对触发时间（s）",
                    ylabel="面积（µm²）")
        _save(fig, output_path)


def plot_spectrum(output_path, frequency, amplitude_um, peak_hz=None):
    """Save particle X/Y amplitudes and their two-dimensional amplitude norm."""
    frequency = np.asarray(frequency, dtype=float)
    amplitude_um = np.asarray(amplitude_um, dtype=float)

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
        if len(frequency) == 0:
            for ax, title in zip(axes, ("颗粒位移低频频谱", "颗粒位移完整单边频谱")):
                ax.text(.5, .5, "没有可用频谱数据", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)
            _save(fig, output_path)
            return
        resolution = frequency[1] - frequency[0] if len(frequency) > 1 else np.nan
        low_limit = min(frequency[-1], max(100., 2 * (peak_hz or 0.)))
        low = frequency <= low_limit
        resultant = np.hypot(amplitude_um[:, 0], amplitude_um[:, 1])
        axes[0].plot(frequency[low], amplitude_um[low, 0], color="#486785", lw=1,
                     label="X 分量")
        axes[0].plot(frequency[low], amplitude_um[low, 1], color="#28786d", lw=1,
                     label="Y 分量")
        axes[0].plot(frequency[low], resultant[low], color="#ae3241", lw=1.5,
                     label="二维合幅 √(X²+Y²)")
        axes[0].set(title="颗粒位移低频频谱", xlabel="频率（Hz）", ylabel="位移幅度（µm）")
        axes[0].grid(axis="y", alpha=.2)
        positive = frequency > 0
        if positive.any():
            for axis, label, line in ((0, "X 分量", "-"), (1, "Y 分量", "--")):
                valid = positive & (amplitude_um[:, axis] > 0)
                axes[1].semilogx(frequency[valid], amplitude_um[valid, axis],
                                 line, lw=1, label=label)
            axes[1].set_yscale("log")
            axes[1].set_xlim(left=frequency[positive].min())
            axes[1].grid(which="both", axis="both", alpha=.2)
        else:
            axes[1].text(.5, .5, "没有可用的正频率颗粒频谱", ha="center", va="center",
                         transform=axes[1].transAxes)
        axes[1].set(title="颗粒位移完整单边频谱", xlabel="频率（Hz，对数轴）",
                    ylabel="幅度（µm，对数轴）")
        if peak_hz is not None and np.isfinite(peak_hz) and peak_hz > 0:
            axes[1].axvline(peak_hz, color="#ae3241", ls=":", lw=1,
                            label=f"主峰 {peak_hz:.2f} Hz")
        note = f"频率分辨率 {resolution:.3g} Hz"
        if peak_hz is None:
            note += "；没有满足条件的局部谱峰"
        axes[0].text(.01, .97, note, transform=axes[0].transAxes,
                     ha="left", va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        axes[0].legend(frameon=True, framealpha=.85, loc="upper right")
        axes[1].legend(frameon=True, framealpha=.85, loc="lower left")
        _save(fig, output_path)


def plot_vibration(output_path, time, displacement_um, rms_um, event_time=None,
                   window_seconds=.020):
    """Save particle displacement and local vibration RMS before fast motion."""
    time = np.asarray(time)
    displacement_um = np.asarray(displacement_um)
    rms_um = np.asarray(rms_um)
    display = np.ones(len(time), dtype=bool)
    if event_time is not None:
        display &= time < event_time
    display_time = time[display]
    display_displacement = displacement_um[display]
    display_rms = rms_um[display]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
        if len(display_time):
            axes[0].plot(display_time, display_displacement[:, 0], lw=.9, label="X")
            axes[0].plot(display_time, display_displacement[:, 1], lw=.9, label="Y")
            axes[0].legend(frameon=False)
        if len(display_time) and len(display_rms):
            axes[1].plot(display_time, display_rms, color="#ae3241", lw=.9, label="二维 RMS")
            axes[1].legend(frameon=False)
        else:
            axes[1].text(.5, .5, "没有可用振动数据", ha="center", va="center",
                         transform=axes[1].transAxes)
        axes[0].set_ylabel("相对首帧位移（µm）")
        axes[0].set_title("快速移动前的质心位移")
        window_ms = window_seconds * 1000
        window_label = f"{window_ms:g} ms"
        axes[1].set(xlabel="相对触发时间（s）",
                    ylabel=f"{window_label} 去趋势 RMS（µm）",
                    title=f"质心局部波动（{window_label} 去线性趋势）")
        if event_time is not None:
            for ax in axes:
                ax.axvline(event_time, color="#ae3241", ls="--", lw=.8)
            if len(display_time):
                axes[1].set_xlim(float(display_time.min()), float(event_time))
        _save(fig, output_path)


def plot_fast_motion(output_path, time, centroid_um, velocity):
    """Save the centroid and speed of the detected fast-motion segment."""
    time = np.asarray(time)
    centroid_um = np.asarray(centroid_um)
    velocity = np.asarray(velocity)
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
        if len(time):
            axes[0].plot(time, centroid_um[:, 0], label="X", lw=1)
            axes[0].plot(time, centroid_um[:, 1], label="Y", lw=1)
            speed = np.linalg.norm(velocity, axis=1)
            axes[1].plot(time, speed, color="#ae3241", lw=1, label="合速度")
            axes[1].plot(time, velocity[:, 0], "--", color="#486785", lw=.8, label="Vx")
            axes[1].plot(time, velocity[:, 1], "--", color="#28786d", lw=.8, label="Vy")
            axes[0].legend(frameon=False)
            axes[1].legend(frameon=False)
        else:
            for ax in axes:
                ax.text(.5, .5, "未检出快速移动", ha="center", va="center",
                        transform=ax.transAxes)
        axes[0].set_ylabel("质心（µm）")
        axes[1].set(xlabel="相对触发时间（s）", ylabel="速度（mm/s）")
        axes[0].set_title("颗粒快速移动")
        _save(fig, output_path)


def plot_flow(output_path, time, flow, fit, trigger, event_time=None,
              event_flow=None, event_label="快速移动", gas_speed=None):
    """Save the MFC flow curve, ramp fit, trigger and event annotations."""
    time = np.asarray(time)
    flow = np.asarray(flow)
    start, slope, intercept, trigger_flow = fit
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.plot(time, flow, ".-", color="#486785", ms=2.5, lw=.8, label="MFC 流量")
        fit_time = np.array([start, 0.])
        ax.plot(fit_time, intercept + slope * fit_time, "--", color="#d17a24",
                lw=2, label=f"拟合斜率 {slope:.3f} SLM/s")
        ax.axvline(0, color=".35", ls=":", lw=.9,
                   label=f"触发 {trigger_flow:.2f} SLM")
        if event_time is not None and event_flow is not None:
            ax.scatter(event_time, event_flow, color="#ae3241", zorder=4)
            ax.annotate(f"{event_label}（插值）  {event_time:+.4f} s\n"
                        f"{event_flow:.2f} SLM",
                        (event_time, event_flow), xytext=(18, -26), va="top",
                        textcoords="offset points", color="#952436",
                        arrowprops={"arrowstyle": "-", "color": "#ae3241"},
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        trigger_text = trigger.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        gas_text = "未计算" if gas_speed is None else f"{gas_speed:.4f} m/s"
        info = (f"触发时刻 {trigger_text}\n"
                f"触发流量 {trigger_flow:.2f} SLM\n"
                f"触发流速 {gas_text}\n"
                f"上升斜率 {slope:.3f} SLM/s")
        ax.text(.02, .97, info, transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": ".7"})
        ax.set(xlabel="相对触发时间（s）", ylabel="流量（SLM）",
               title="MFC 流量随时间变化")
        ax.legend(frameon=True, framealpha=.85, loc="lower right")
        _save(fig, output_path)


def plot_ringdown(output_path, time_s, position_um, fitted_um, baseline_um,
                  envelope_um, fit_results, title):
    """Save per-axis ringdown positions, envelopes, fits, and residuals."""
    time_ms = np.asarray(time_s, dtype=float) * 1000
    position_um = np.asarray(position_um, dtype=float)
    fitted_um = np.asarray(fitted_um, dtype=float)
    baseline_um = np.asarray(baseline_um, dtype=float)
    envelope_um = np.asarray(envelope_um, dtype=float)
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True,
                                 layout="constrained")
        labels = ("X", "Y")
        for axis, label in enumerate(labels):
            raw = position_um[:, axis] - position_um[0, axis]
            fitted = fitted_um[:, axis] - position_um[0, axis]
            residual = position_um[:, axis] - fitted_um[:, axis]
            centered = position_um[:, axis] - baseline_um[:, axis]
            axes[0, axis].plot(time_ms, raw, color="#486785", lw=1,
                               label="原始位置")
            axes[0, axis].plot(time_ms, fitted, color="#ae3241", lw=1.2,
                               label="拟合")
            result = fit_results[axis]
            axes[0, axis].set_title(
                f"{label}: fd={result['frequency_hz']:.3g} Hz, "
                f"τ={result['tau_s'] * 1000:.3g} ms, R²={result['r2']:.3f}")
            axes[0, axis].legend(frameon=False, fontsize=8)
            axes[1, axis].plot(time_ms, centered, color="#486785", lw=1,
                               label="位置−基线")
            axes[1, axis].plot(time_ms, envelope_um[:, axis], "--",
                               color="#d17a24", lw=.9, label="+包络")
            axes[1, axis].plot(time_ms, -envelope_um[:, axis], "--",
                               color="#d17a24", lw=.9, label="−包络")
            axes[1, axis].legend(frameon=False, fontsize=8)
            axes[2, axis].plot(time_ms, residual, color="#28786d", lw=1)
            axes[2, axis].axhline(0, color=".5", lw=.6)
            axes[2, axis].set_xlabel("片段内时间（ms）")
            axes[0, axis].set_ylabel("位移（µm）")
            axes[1, axis].set_ylabel("位移（µm）")
            axes[2, axis].set_ylabel("残差（µm）")
        fig.suptitle(title)
        _save(fig, output_path)


def plot_trajectory_wavelet(output_path, time_s, position_um, frequency_hz,
                            power, edge_mask, title, power_max):
    """Save the time-frequency map of a continuous tracking interval."""
    time_s = np.asarray(time_s, dtype=float)
    position_um = np.asarray(position_um, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    power = np.asarray(power, dtype=float)
    edge_mask = np.asarray(edge_mask, dtype=bool)
    vmax = float(power_max)
    # 强烈的离开运动会抬高全局上限，对数色标保留较弱振动的可见度。
    power_norm = LogNorm(vmin=vmax / 1e6, vmax=vmax, clip=True) if vmax > 0 else Normalize(0, 1)

    # 长轨迹仅在绘图时按时间像素取最大值，保留短暂峰值。数值文件仍为逐帧功率。
    step = max(1, int(np.ceil(len(time_s) / 1600)))
    if step > 1:
        starts = np.arange(0, len(time_s), step)
        stops = np.r_[starts[1:] - 1, len(time_s) - 1]
        plot_time = (time_s[starts] + time_s[stops]) / 2
        plot_power = np.maximum.reduceat(power, starts, axis=2)
        plot_edge = np.logical_or.reduceat(edge_mask, starts, axis=1)
    else:
        plot_time, plot_power, plot_edge = time_s, power, edge_mask

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                 layout="constrained",
                                 gridspec_kw={"height_ratios": (1, 1.35, 1.35)})
        centered = position_um - position_um[0]
        axes[0].plot(time_s, centered[:, 0], lw=.8, color="#486785", label="X")
        axes[0].plot(time_s, centered[:, 1], lw=.8, color="#28786d", label="Y")
        axes[0].set_ylabel("相对首帧位移（µm）")
        axes[0].legend(frameon=False, ncol=2, loc="upper right")

        meshes = []
        for axis, label in enumerate(("X", "Y"), start=1):
            mesh = axes[axis].pcolormesh(
                plot_time, frequency_hz, plot_power[axis - 1], shading="auto",
                norm=power_norm, cmap="viridis")
            meshes.append(mesh)
            axes[axis].set_yscale("log")
            axes[axis].set_ylabel(f"{label} 频率（Hz，对数轴）")
            axes[axis].contourf(plot_time, frequency_hz, plot_edge.astype(float),
                                levels=[.5, 1.5], colors="#808080", alpha=.18,
                                hatches=["///"])

        axes[-1].set(xlabel="相对触发时间（s）", xlim=(time_s[0], time_s[-1]))
        color_label = "尺度归一化小波功率（对数色标）" if vmax > 0 else "尺度归一化小波功率"
        fig.colorbar(meshes[0], ax=axes[1:], label=color_label)
        fig.suptitle(f"{title}\n斜线区域：连续段的边缘影响参考区")
        _save(fig, output_path)


def plot_vibration_candidates(output_path, time_s, position_um, rms_um,
                              start_threshold_um, end_threshold_um,
                              candidate_ranges, rms_window_s):
    """Save a review plot for candidate vibration intervals selected by RMS."""
    time_s = np.asarray(time_s, dtype=float)
    position_um = np.asarray(position_um, dtype=float)
    rms_um = np.asarray(rms_um, dtype=float)
    valid = np.isfinite(position_um).all(axis=1)
    reference = position_um[np.flatnonzero(valid)[0]] if valid.any() else np.zeros(2)
    centered = position_um - reference
    window_ms = rms_window_s * 1000

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                 layout="constrained")
        axes[0].plot(time_s, centered[:, 0], lw=.8, color="#486785", label="X")
        axes[0].plot(time_s, centered[:, 1], lw=.8, color="#28786d", label="Y")
        axes[0].set_ylabel("相对首个有效点位移（µm）")
        axes[0].legend(frameon=False, ncol=2, loc="upper right")

        for number, start, end in candidate_ranges:
            axes[0].axvspan(start, end, color="#ae3241", alpha=.12, lw=0)
            axes[0].text((start + end) / 2, .96, str(number),
                         transform=axes[0].get_xaxis_transform(), ha="center",
                         va="top", color="#952436", fontsize=6, rotation=90)

        axes[1].plot(time_s, rms_um, color="#ae3241", lw=.9, label="局部振动 RMS")
        axes[1].axhline(start_threshold_um, color="#d17a24", ls="--", lw=1,
                        label=f"开始阈值 {start_threshold_um:g} µm")
        axes[1].axhline(end_threshold_um, color="#28786d", ls=":", lw=1,
                        label=f"结束阈值 {end_threshold_um:g} µm")
        axes[1].set(xlabel="相对触发时间（s）",
                    ylabel="振动 RMS（µm，对数轴）", yscale="log",
                    title=f"候选振动区间（RMS 窗口 {window_ms:g} ms）")
        axes[1].legend(frameon=False, loc="upper right")
        fig.suptitle("完整轨迹的振动候选段复核")
        _save(fig, output_path)


def plot_event_distributions(output_path, energy, onset_intervals_s, quiet_gaps_s):
    """Plot distributions of cumulative squared displacement and event intervals."""
    series = (
        (energy, "RMS 平方积分（µm²·s）"),
        (np.asarray(onset_intervals_s) * 1000, "相邻事件起点间隔（ms）"),
        (np.asarray(quiet_gaps_s) * 1000, "上一段结束至本段开始（ms）"),
    )
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for index, (ax, (values, label)) in enumerate(zip(axes, series)):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                bins = min(20, max(1, int(np.ceil(np.sqrt(len(values))))))
                if index == 0 and 0 < values.min() < values.max():
                    bins = np.geomspace(values.min(), values.max(), bins + 1)
                    ax.set_xscale("log")
                    label = "RMS 平方积分（µm²·s，对数轴）"
                ax.hist(values, bins=bins, color="#486785", edgecolor="white")
            else:
                ax.text(.5, .5, "暂无数据", transform=ax.transAxes, ha="center", va="center")
            ax.set(xlabel=label, ylabel="数量", title=f"{len(values)} 个样本")
        fig.suptitle("候选振动强度与间隔分布")
        _save(fig, output_path)


def plot_ringdown_selection(output_path, video_frames, position_um, fit_range,
                            title):
    """Plot an approximate ringdown window and its selected fit interval."""
    video_frames = np.asarray(video_frames, dtype=float)
    position_um = np.asarray(position_um, dtype=float)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        for axis, label, color in ((0, "X", "#486785"),
                                   (1, "Y", "#28786d")):
            values = position_um[:, axis]
            finite = np.isfinite(values)
            if finite.any():
                first = values[np.flatnonzero(finite)[0]]
                centered = values - first
                ax.plot(video_frames, np.where(finite, centered, np.nan),
                        color=color, lw=.9, label=label)

        if fit_range is not None:
            start, end = fit_range
            ax.axvspan(float(start), float(end), color="#ae3241", alpha=.14,
                       lw=0, label=f"实际拟合帧 {int(start)}–{int(end)}")

        ax.set(xlabel="原视频帧号", ylabel="相对首个有效点位移（µm）",
               title=title)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False)
        _save(fig, output_path)
