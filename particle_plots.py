"""Plots for particle motion, vibration and MFC flow."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
        if centroid_um.ndim == 2 and centroid_um.shape[1] == 2:
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
        if amplitude_um.ndim == 2 and amplitude_um.shape == (len(frequency), 2):
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
        if amplitude_um.ndim == 2 and amplitude_um.shape == (len(frequency), 2) and positive.any():
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
        if len(display_time) and display_displacement.ndim == 2 and display_displacement.shape[1] == 2:
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
        if hasattr(trigger, "strftime"):
            trigger_text = trigger.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            trigger_text = str(trigger)
            if "." in trigger_text:
                dot = trigger_text.rfind(".")
                trigger_text = trigger_text[:dot + 4]
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
