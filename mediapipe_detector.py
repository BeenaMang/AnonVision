# mediapipe_detector.py
# Minimal wrapper so AnonVision can optionally use MediaPipe.
# If mediapipe isn't installed or fails to import, the GUI should continue with YuNet only.

from typing import List, Tuple
import numpy as np

class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except Exception as e:
            raise ImportError(
                "mediapipe is not installed or not supported on this Python version. "
                "Install with: pip install mediapipe"
            ) from e
        self._mp = mp
        # model_selection: 0=short-range, 1=full-range
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=float(min_detection_confidence)
        )

    def detect_faces(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[tuple]]:
        """Returns (input_image_bgr, [(x,y,w,h), ...])"""
        h, w = image_bgr.shape[:2]
        image_rgb = image_bgr[:, :, ::-1]
        res = self._fd.process(image_rgb)

        boxes: List[tuple] = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                ww = max(0, int(bb.width * w))
                hh = max(0, int(bb.height * h))
                if ww > 0 and hh > 0:
                    boxes.append((x, y, ww, hh))
        return image_bgr, boxes
