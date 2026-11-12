"""从已导出的质心轨迹筛选候选振动，供人工检查后选择衰减拟合区间。"""

import csv
import json
from pathlib import Path

import numpy as np

from particle_data import load_measurements, write_numeric_csv
from particle_overlay import export_overlay_clips
from particle_plots import plot_vibration_candidates
from particle_postprocess import motion_metrics


def find_vibration_intervals(rms_um, fps, start_threshold_um, end_threshold_um,
                             min_duration_s, quiet_duration_s, merge_gap_s,
                             padding_before_s, padding_after_s):
    """迟滞阈值筛选，返回含两端的行索引；NaN 是不能跨越的边界。"""
    if not 0 < end_threshold_um < start_threshold_um:
        raise ValueError("RMS 阈值须满足 0 < 结束阈值 < 开始阈值。")
    rms = np.asarray(rms_um, dtype=float)
    finite = np.isfinite(rms)
    min_count = max(1, round(min_duration_s * fps))
    quiet_count = max(1, round(quiet_duration_s * fps))
    merge_count = round(merge_gap_s * fps)
    before, after = round(padding_before_s * fps), round(padding_after_s * fps)
    intervals = []
    start, high_count, low_count = None, 0, 0
    for index, value in enumerate(np.r_[rms, np.nan]):
        if not np.isfinite(value):
            if start is not None:
                intervals.append((start, index - low_count - 1))
            start, high_count, low_count = None, 0, 0
        elif start is None:
            high_count = high_count + 1 if value >= start_threshold_um else 0
            if high_count >= min_count:
                start = index - high_count + 1
        else:
            low_count = low_count + 1 if value < end_threshold_um else 0
            if low_count >= quiet_count:
                intervals.append((start, index - low_count))
                start, high_count, low_count = None, 0, 0

    merged = []
    for start, end in intervals:
        if (merged and start - merged[-1][1] - 1 <= merge_count
                and finite[merged[-1][1] + 1:start].all()):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    padded = []
    for start, end in merged:
        left, right = max(0, start - before), min(len(rms) - 1, end + after)
        missing_left = np.flatnonzero(~finite[left:start])
        missing_right = np.flatnonzero(~finite[end + 1:right + 1])
        if len(missing_left):
            left += int(missing_left[-1]) + 1
        if len(missing_right):
            right = end + int(missing_right[0])
        padded.append((left, right))
    return padded


def extract_vibration_events(measurement_dir, overlay_path, output_dir,
                              rms_window_s=.004, start_threshold_um=1.5,
                              end_threshold_um=.75, min_duration_s=.001,
                              quiet_duration_s=.002, merge_gap_s=.003,
                              padding_before_s=.002, padding_after_s=.004):
    """RMS 用采集帧率计算；裁视频用原视频帧号，保留标注视频的播放帧率。"""
    info, rows = load_measurements(measurement_dir)
    fps = float(info["capture_fps"])
    frames = np.array([int(row["video_frame"]) for row in rows])
    source_frames = np.array([int(row["source_frame"]) for row in rows])
    time = np.array([float(row["time_from_trigger_s"]) for row in rows])
    position = np.array([[float(row["centroid_x_um"] or "nan"),
                          float(row["centroid_y_um"] or "nan")] for row in rows])
    valid = np.array([row["tracking_valid"] == "1" for row in rows])
    position[~valid] = np.nan
    if np.any(np.diff(frames) != 1):
        raise ValueError("测量表的视频帧号须连续；失踪帧应保留空记录。")
    _, rms = motion_metrics(position, fps, vibration_window_s=rms_window_s)
    intervals = find_vibration_intervals(
        rms, fps, start_threshold_um, end_threshold_um, min_duration_s,
        quiet_duration_s, merge_gap_s, padding_before_s, padding_after_s)
    segments = [(number, int(frames[start]), int(frames[end]))
                for number, (start, end) in enumerate(intervals, start=1)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 表中范围和视频文件名一致，包含前后余量；自由衰减拟合仍需看视频再选尾段。
    with (output_dir / "segments.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event_id", "start_frame", "end_frame"])
        writer.writerows(segments)
    lines = ["segments = [", *(f"    ({number}, {start}, {end})," for number, start, end in segments), "]"]
    (output_dir / "segments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_numeric_csv(output_dir / "rms.csv",
                      ["source_frame", "video_frame", "time_from_trigger_s", "vibration_rms_um"],
                      np.column_stack((source_frames, frames, time, rms)), decimal_places=None)
    ranges = [(number, float(time[start]), float(time[end]))
              for number, (start, end) in enumerate(intervals, start=1)]
    plot_vibration_candidates(output_dir / "candidates.png", time, position, rms,
                              start_threshold_um, end_threshold_um, ranges, rms_window_s)
    settings = {
        "measurement_dir": str(Path(measurement_dir).resolve()),
        "overlay_path": str(Path(overlay_path).resolve()), "capture_fps": fps,
        "rms_window_s": rms_window_s, "start_threshold_um": start_threshold_um,
        "end_threshold_um": end_threshold_um, "min_duration_s": min_duration_s,
        "quiet_duration_s": quiet_duration_s, "merge_gap_s": merge_gap_s,
        "padding_before_s": padding_before_s, "padding_after_s": padding_after_s,
        "start_video_frame": int(frames[0]), "end_video_frame": int(frames[-1]),
        "candidate_count": len(segments), "frame_numbering": "1-based video_frame, inclusive",
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    export_overlay_clips(overlay_path, output_dir / "clips", segments,
                         first_video_frame=int(info["start_video_frame"]))
    print(f"筛出 {len(segments)} 个候选波动段；结果：{output_dir}")
    return segments
