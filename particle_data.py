"""测量文件的读写与相机/MFC 时间基准，不依赖视频解码或后处理算法。"""

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def read_flow(flow_path):
    with open(flow_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    times = [datetime.strptime(row["Time"], "%Y/%m/%d %H:%M:%S:%f") for row in rows]
    setpoint = np.array([float(row["SetPoint"]) for row in rows])
    readflow = np.array([float(row["ReadFlow"]) for row in rows])
    return times, setpoint, readflow


def match_trigger(log_path, metadata, first_frame, start, end):
    pattern = re.compile(r"^\[([^|]+)\|INFO\]\[" + re.escape(metadata["ip"]) + r"\]SoftTrigger OK\.")
    triggers = []
    for line in Path(log_path).read_text(encoding="utf-8-sig").splitlines():
        match = pattern.match(line)
        if match:
            trigger = datetime.fromisoformat(match[1])
            if start <= trigger <= end:
                triggers.append(trigger)
    if not triggers:
        raise ValueError("MFC 记录期间没有找到这台相机的 SoftTrigger OK。")
    if len(triggers) == 1:
        return triggers[0]
    # 仅当一个 MFC 文件里有多次触发时，用相机首帧的整秒区分它们。
    camera_start = datetime.fromtimestamp(metadata["timestamp"], timezone.utc).replace(tzinfo=None)
    estimated = camera_start + timedelta(seconds=(metadata["key_frame_no"] - first_frame)
                                         / metadata["frame_rate"])
    return min(triggers, key=lambda t: abs(t - estimated))


def write_numeric_csv(path, columns, rows, decimal_places=8):
    """空值写为空白；decimal_places=None 保留 float64 往返所需的完整精度。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            values = []
            for value in row:
                if not np.isfinite(value):
                    values.append("")
                elif decimal_places is None:
                    values.append(repr(float(value)))
                else:
                    text = f"{value:.{decimal_places}f}"
                    values.append(text.rstrip("0").rstrip(".") if decimal_places > 0 else text)
            writer.writerow(values)


def load_measurements(measurement_dir):
    """读取原始测量元数据与逐帧记录；不访问元数据中记载的原文件路径。"""
    measurement_dir = Path(measurement_dir)
    info = json.loads((measurement_dir / "measurement_info.json").read_text(encoding="utf-8"))
    if info["format_version"] != 1:
        raise ValueError("不支持此测量文件版本。")
    with (measurement_dir / "frames.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return info, rows


def load_flow_samples(measurement_dir):
    """读取已固定到触发时间基准的完整 MFC 原始采样记录。"""
    values = np.loadtxt(Path(measurement_dir) / "flow_samples.csv",
                        delimiter=",", skiprows=1, ndmin=2)
    return values[:, 0], values[:, 1], values[:, 2]

