from pathlib import Path
from particle_motion import analyze_video

workspace = Path(__file__).resolve().parents[1]
experiment_dir = workspace / "261110"

# 修改下面这几行即可切换一次分析的视频。
video_path = experiment_dir / "1-1-45000.mp4"
flow_path = experiment_dir / "1.csv"
calibration_path = experiment_dir / "calibration" / "calibration.json"
log_path = experiment_dir / "oplog" / "revealer_2026-11-10.log"
output_dir = None  # 默认：视频旁的 序号/measurements；保留多次测量时可指定不同目录。

# 视频内的帧序号，从 1 开始，包含起止两帧；None 表示从开头/处理到结尾。
start_frame = None
end_frame = None

# 标注视频的播放帧率；None 沿用输入视频的播放帧率，不影响分析时间轴。
annotated_video_fps = None


if __name__ == "__main__":
    analyze_video(
        video_path=video_path,
        flow_path=flow_path,
        calibration_path=calibration_path,
        log_path=log_path,
        output_dir=output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        annotated_video_fps=annotated_video_fps,
    )
