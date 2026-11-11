from pathlib import Path

from particle_postprocess import postprocess_measurements


workspace = Path(__file__).resolve().parents[1]
experiment_dir = workspace / "261110"
measurement_dir = experiment_dir / "1" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "baseline"

event_label = "快速移动"
event_frame = None
full_scale_slm = 500.
channel_width_mm, channel_height_mm = 27., 12.
velocity_window_s = .002
vibration_window_s = .020


if __name__ == "__main__":
    postprocess_measurements(
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        event_label=event_label,
        event_frame=event_frame,
        full_scale_slm=full_scale_slm,
        channel_width_mm=channel_width_mm,
        channel_height_mm=channel_height_mm,
        velocity_window_s=velocity_window_s,
        vibration_window_s=vibration_window_s,
    )
