from pathlib import Path
from particle_postprocess import postprocess_measurements

workspace = Path(r"D:/revealer/Workspace/sus")
experiment_dir = workspace / "261112"
measurement_dir = experiment_dir / "7" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "baseline"

event_label = "快速移动"
event_frame = None
full_scale_slm = 500.
start_flow_slm = 5.  # 流量上升段的起始流量，单位 SLM。
channel_width_mm, channel_height_mm = 27., 12.
velocity_window_s = .002
vibration_window_s = .020
wavelet_frequency_range_hz = (100., 2400.)  # 小波上限须低于采集帧率的一半。
wavelet_frequency_count = 160

# 读取测量 → 常规运动与流量汇总 → 对各连续有效轨迹计算小波。
if __name__ == "__main__":
    postprocess_measurements(
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        event_label=event_label,
        event_frame=event_frame,
        full_scale_slm=full_scale_slm,
        start_flow_slm=start_flow_slm,
        channel_width_mm=channel_width_mm,
        channel_height_mm=channel_height_mm,
        velocity_window_s=velocity_window_s,
        vibration_window_s=vibration_window_s,
        wavelet_frequency_range_hz=wavelet_frequency_range_hz,
        wavelet_frequency_count=wavelet_frequency_count,
    )
