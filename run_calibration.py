"""直接在 IDE 运行本文件；修改下方路径和参数即可。"""

from pathlib import Path

from checkerboard_grid import calibrate_checkerboard


workspace = Path(r"D:/revealer/Workspace/sus")
experiment_dir = workspace / "261112"
image_path = experiment_dir / "192.168.8.52" / "192.168.8.52_20261112_212401_831455.bmp"
square_size_mm = 0.5
inner_corners = (4, 3)  # 内角点的列数、行数（不是方格数）
output_dir = experiment_dir / "calibration"


if __name__ == "__main__":
    calibrate_checkerboard(
        image_path=image_path,
        square_size_mm=square_size_mm,
        inner_corners=inner_corners,
        output_dir=output_dir,
    )
