# 颗粒振动分析脚本

## 运行顺序

1. **棋盘格标定** · 运行 `run_calibration.py`，得到比例尺文件 `calibration.json` 和检查图 `calibration.png`。已有适用的标定结果时，可以直接在下一步指定该 JSON。
2. **视频测量** · 运行 `analyze_video.py`。设置视频、标定 JSON、MFC CSV 和相机日志，视频旁需有同名 JSON。可用 `start_frame`、`end_frame` 指定导出范围，两者均为 `None` 时处理整个视频。结果默认保存在 `<序号>/measurements/`，包括逐帧数据、轮廓、流量记录和带标记视频。
3. **按分析目的运行后处理** · 各入口顶部的 `measurement_dir` 应指向上一步实际生成的测量目录。
   - **位移、速度、频谱、流量和整段小波** · 运行 `postprocess_particles.py`，结果默认写入 `postprocess/baseline/`。
   - **从长视频筛选振动** · 运行 `extract_vibration_events.py`，在 `postprocess/vibration_candidates/` 查看带标记片段、候选清单及强度与间隔分布图。用 `segments.csv` 末列 `energy_rank` 查找累计位移波动较强的候选。
   - **逐段衰减拟合** · 查看候选片段或手动选段，把 `(编号, 起始帧, 终止帧)` 填入 `fit_particle_vibrations.py` 的 `segments`。拟合区间通常从激励结束后开始，截止于下一次激励前。拟合参数、曲线、包络和残差默认写入 `postprocess/ringdown/`。


选段、候选表和视频文件名使用原视频内的 `video_frame`，从 1 开始，包含起止帧。持续时间和频率按采集帧率计算。

小波按连续有效跟踪区间分别计算和出图，保留原帧号和触发时间。拟合只要求所选帧连续有效，可以直接在同一个测量目录上运行。

## 入口及其主要函数

| 直接运行的脚本 | 支撑模块与入口函数 | 主要支撑函数 |
| --- | --- | --- |
| [run_calibration.py](run_calibration.py) | [checkerboard_grid.py](checkerboard_grid.py) 中的 `calibrate_checkerboard()` | `_grid_line_positions()` 定位网格线，供主函数按格距计算比例尺。 |
| [analyze_video.py](analyze_video.py) | [particle_motion.py](particle_motion.py) 中的 `analyze_video()` | `selected_frame_bounds()` 确定处理范围。`track_video()` 调用 `first_particle()`、`particle_background()`、`particle_measurement()` 初始化并逐帧测量。 |
| [postprocess_particles.py](postprocess_particles.py) | [particle_postprocess.py](particle_postprocess.py) 中的 `postprocess_measurements()` | `motion_metrics()` 计算速度和位移 RMS，`departure_index()` 查找快速移动起点，`centroid_spectrum()` 计算频谱，`fit_ramp()` 拟合流量斜坡，`write_summary()` 导出 Excel 汇总，`export_trajectory_wavelet()` 输出小波数据和图。 |
| [extract_vibration_events.py](extract_vibration_events.py) | [particle_events.py](particle_events.py) 中的 `extract_vibration_events()` | 复用 `motion_metrics()` 计算 RMS，再由 `find_vibration_intervals()` 按双阈值和持续时间筛选候选。 |
| [fit_particle_vibrations.py](fit_particle_vibrations.py) | [particle_ringdown.py](particle_ringdown.py) 中的 `analyze_vibration_segments()` | `fit_ringdown()` 对各段 X、Y 分别拟合振动频率和衰减时间。 |

## 共用模块

下列模块由入口导入，参数集中在上表的入口脚本顶部。

| 模块 | 职责 |
| --- | --- |
| [particle_data.py](particle_data.py) | `read_flow()`、`match_trigger()` 读取流量和匹配触发，`load_measurements()`、`load_flow_samples()` 读取导出数据，`write_numeric_csv()` 写数值表。 |
| [particle_overlay.py](particle_overlay.py) | `draw_tracking_overlay()` 绘制帧号、质心和边缘，`export_overlay_clips()` 按帧拆分标注视频。 |
| [particle_wavelet.py](particle_wavelet.py) | `morlet_power()` 计算复 Morlet 小波功率，`export_trajectory_wavelet()` 导出各有效段，所有段共用色标。 |
| [particle_plots.py](particle_plots.py) | `plot_measurements()` 画测量图。`plot_vibration()`、`plot_spectrum()`、`plot_fast_motion()`、`plot_flow()` 画常规后处理图。`plot_vibration_candidates()`、`plot_event_distributions()`、`plot_ringdown()`、`plot_trajectory_wavelet()` 分别画候选、统计分布、拟合和小波图。 |

更多参数及输出字段见 [使用说明.txt](../使用说明.txt)。
