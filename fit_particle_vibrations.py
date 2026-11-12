"""完整导出轨迹统一绘制小波图，手动指定帧范围分别拟合衰减振动。"""

from pathlib import Path

from particle_ringdown import analyze_vibration_segments

workspace = Path(__file__).resolve().parents[1]
measurement_dir = workspace / "261110" / "1" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "ringdown"

# (振动编号, 起始帧, 终止帧)：原视频 1-based 帧号，包含两端；仅控制拟合和局部展示。
# 小波始终使用 measurement_dir 中导出的完整连续轨迹，不受下面的分段范围限制。
# 以下是现有轨迹的示例尾段；起点应放在激励结束后，终点避开再次激励。
segments = [
    (1, 26000, 26200),
    (2, 43000, 43265),
]

frequency_range_hz = (100., 2500.)  # 拟合搜索范围和小波显示范围，须低于采集帧率的一半。
tau_range_s = (.0001, .1)          # 振幅衰减时间常数的搜索范围（秒）。
wavelet_frequency_count = 160     # 对数分布的频率点数，只控制小波图的频率采样。

if __name__ == "__main__":
    analyze_vibration_segments(
        measurement_dir=measurement_dir,
        output_dir=output_dir,
        segments=segments,
        frequency_range_hz=frequency_range_hz,
        tau_range_s=tau_range_s,
        wavelet_frequency_count=wavelet_frequency_count,
    )
