"""给定候选范围，自动选择衰减尾段并逐段拟合。"""

from pathlib import Path

from particle_ringdown import analyze_vibration_segments

workspace = Path(__file__).resolve().parents[1]
measurement_dir = workspace / "261110" / "1" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "ringdown"

# (振动编号, 搜索起始帧, 搜索终止帧)：原视频 1-based 帧号，包含两端。
# 每个范围围住一次待分析的响应。已有精确区间时可关闭自动选段。
segments = [
    (3, 3711, 3869),
]

frequency_range_hz = (100., 2500.)  # 拟合搜索范围，须低于采集帧率的一半。
tau_range_s = (.0001, .1)          # 振幅衰减时间常数的搜索范围（秒）。
auto_select_window = True
min_cycles = 3.                   # 自动选段要求的最少振动周期数。
min_oscillation_r2 = .75          # 相对于线性漂移模型，振荡须解释至少 75% 的剩余波动。
min_amplitude_um = .25            # 自动选段接受的最小拟合起始振幅。

if __name__ == "__main__":
    analyze_vibration_segments(
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        segments=segments,
        frequency_range_hz=frequency_range_hz,
        tau_range_s=tau_range_s,
        auto_select_window=auto_select_window,
        min_cycles=min_cycles,
        min_oscillation_r2=min_oscillation_r2,
        min_amplitude_um=min_amplitude_um,
    )
