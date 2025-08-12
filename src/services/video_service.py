from typing import Generator, Tuple

import cv2
import numpy as np


def process_video_frames(
    video_path: str,
) -> Generator[Tuple[bool, np.ndarray], None, None]:
    """Process video and yield frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield False, np.array([])
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield True, frame
    cap.release()
