"""手动指定每次振动的原视频帧范围，分别拟合并绘制小波图。"""

from pathlib import Path

from particle_ringdown import analyze_vibration_segments

workspace = Path(__file__).resolve().parents[1]
measurement_dir = workspace / "261110" / "1" / "measurements"
output_dir = measurement_dir.parent / "postprocess" / "ringdown"

# (振动编号, 起始帧, 终止帧)：原视频 1-based 帧号，包含两端。
# 以下是现有轨迹的示例尾段；起点应放在激励结束后，终点避开再次激励。
segments = [
    (1, 3530, 3610),
    (2, 3760, 3830),
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
