"""手动指定帧范围分别拟合衰减振动。"""

from pathlib import Path

from particle_ringdown import analyze_vibration_segments

workspace = Path(__file__).resolve().parents[1]
measurement_dir = workspace / "261110" / "1" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "ringdown"

# (振动编号, 起始帧, 终止帧)：原视频 1-based 帧号，包含两端；仅控制拟合。
# 以下是现有轨迹的示例尾段；起点应放在激励结束后，终点避开再次激励。
segments = [
    (3, 3711, 3869),
]

frequency_range_hz = (100., 2500.)  # 拟合搜索范围，须低于采集帧率的一半。
tau_range_s = (.0001, .1)          # 振幅衰减时间常数的搜索范围（秒）。

if __name__ == "__main__":
    analyze_vibration_segments(
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        segments=segments,
        frequency_range_hz=frequency_range_hz,
        tau_range_s=tau_range_s,
    )
