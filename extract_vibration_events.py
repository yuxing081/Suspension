"""按质心去趋势 RMS 筛选候选波动，并拆分已有标注视频。"""

from pathlib import Path

from particle_events import extract_vibration_events

workspace = Path(r"D:/revealer/Workspace/sus")
measurement_dir = workspace / "261112" / "8" / "measurements"
overlay_path = measurement_dir / "tracking_overlay.mp4"
output_dir = measurement_dir.parent / "postprocess" / "vibration_candidates"

rms_window_s = .004               # 局部去线性趋势窗口，4 ms 约覆盖 700 Hz 的 3 个周期。
start_threshold_um = 1.5          # RMS 开始阈值（µm）；调低会检出更多弱波动。
end_threshold_um = .75            # RMS 结束阈值（µm），须低于开始阈值。
min_duration_s = .001             # 超过开始阈值至少持续 1 ms，忽略单帧尖峰。
quiet_duration_s = .002           # 低于结束阈值持续 2 ms，才认为本段结束。
merge_gap_s = .003                # 相隔不超过 3 ms 的候选段合并，不能跨失踪帧。
padding_before_s = .002           # 视频在检测起点前保留 2 ms。
padding_after_s = .004            # 视频在检测终点后保留 4 ms。

if __name__ == "__main__":
    extract_vibration_events(
        measurement_dir=measurement_dir,
        overlay_path=overlay_path,
        output_dir=output_dir,
        rms_window_s=rms_window_s,
        start_threshold_um=start_threshold_um,
        end_threshold_um=end_threshold_um,
        min_duration_s=min_duration_s,
        quiet_duration_s=quiet_duration_s,
        merge_gap_s=merge_gap_s,
        padding_before_s=padding_before_s,
        padding_after_s=padding_after_s,
    )
