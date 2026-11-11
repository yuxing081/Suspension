"""Draw a non-destructive tracking diagnostic overlay on one video frame."""

import cv2
import numpy as np


def draw_tracking_overlay(
    frame,
    center,
    outline,
    video_frame,
    source_frame,
    time_from_trigger_s,
):
    """Return a BGR uint8 copy of ``frame`` with tracking diagnostics drawn.

    ``outline`` contains unordered ``(x, y)`` boundary pixels, as produced by
    the particle tracker after converting ``np.nonzero`` output to image
    coordinates. Points are painted individually so no artificial connecting
    segments are introduced.
    """
    source = np.asarray(frame)
    if source.ndim == 2:
        output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif source.ndim == 3 and source.shape[2] == 3:
        output = source.copy()
    else:
        raise ValueError("frame must be a grayscale image or a 3-channel BGR image")
    if output.dtype != np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)

    height, width = output.shape[:2]
    detected = False
    if center is not None:
        center_values = np.asarray(center, dtype=float).reshape(-1)
        detected = len(center_values) >= 2 and np.isfinite(center_values[:2]).all()

    status = "detected" if detected else "not detected"
    lines = (
        f"Video frame: {video_frame} | Source frame: {source_frame}",
        f"t = {float(time_from_trigger_s):+.6f} s | {status}",
    )
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0]
                     for line in lines)
    cv2.rectangle(output, (0, 0), (min(width - 1, text_width + 16), min(height - 1, 54)),
                  (20, 20, 20), -1)
    for line, y in zip(lines, (24, 46)):
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    if detected and outline is not None:
        points = np.asarray(outline, dtype=float)
        if points.size:
            points = points.reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                points = np.rint(points).astype(np.int64)
                for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    x = np.clip(points[:, 0] + offset_x, 0, width - 1)
                    y = np.clip(points[:, 1] + offset_y, 0, height - 1)
                    output[y, x] = (0, 255, 0)

    if detected:
        center_values = np.asarray(center, dtype=float).reshape(-1)
        center_x, center_y = np.rint(center_values[:2]).astype(int)
        center_x = int(np.clip(center_x, 0, width - 1))
        center_y = int(np.clip(center_y, 0, height - 1))
        cv2.drawMarker(
            output,
            (center_x, center_y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=13,
            thickness=1,
            line_type=cv2.LINE_8,
        )
        cv2.circle(output, (center_x, center_y), 2, (0, 0, 255), -1, cv2.LINE_8)

    return output
