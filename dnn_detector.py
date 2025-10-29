# dnn_detector.py (YuNet-first, CPU-only; Haar fallback)
from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
import os
import cv2
import numpy as np

BBox = Tuple[int,int,int,int]

_ZOO_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

class DNNFaceDetector:
    """YuNet via cv2.FaceDetectorYN or ONNX; falls back to Haar if needed."""
    def __init__(self, data_dir: str = "data", score_threshold: float = 0.5, nms_threshold: float = 0.3):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_path = str(Path(self.data_dir) / "face_detection_yunet_2023mar.onnx")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

        self.detector_yn = None
        self.net = None
        self.cascade = None

        # Try YuNet binding first
        try:
            if not Path(self.model_path).exists():
                # Best-effort download (user should have internet when running locally)
                import urllib.request
                urllib.request.urlretrieve(_ZOO_URL, self.model_path)
            self.detector_yn = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                5000
            )
        except Exception:
            self.detector_yn = None

        # Fallback: Haar cascade
        if self.detector_yn is None:
            try:
                haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.cascade = cv2.CascadeClassifier(haar)
            except Exception:
                self.cascade = None

        # Force CPU-only / disable OpenCL to avoid cuDNN/OpenCL issues
        try:
            if hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

    def set_score_threshold(self, thr: float):
        self.score_threshold = float(thr)
        if self.detector_yn is not None:
            try:
                self.detector_yn.setScoreThreshold(self.score_threshold)
            except Exception:
                pass

    def detect_faces(self, bgr_img: np.ndarray, confidence_threshold: Optional[float] = None) -> Tuple[np.ndarray, List[BBox]]:
        """Return (input_image, [(x,y,w,h), ...])"""
        if bgr_img is None or bgr_img.size == 0:
            return bgr_img, []

        img = bgr_img
        h, w = img.shape[:2]
        faces: List[BBox] = []
        thr = float(confidence_threshold) if confidence_threshold is not None else self.score_threshold

        if self.detector_yn is not None:
            try:
                self.detector_yn.setInputSize((w, h))
                res = self.detector_yn.detect(img)
                boxes = res[1] if isinstance(res, tuple) and len(res) > 1 else None
                if boxes is not None:
                    # boxes: [x, y, w, h, score, ...]
                    for row in boxes:
                        x, y, ww, hh, score = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
                        if score >= thr and ww > 0 and hh > 0:
                            # clamp box
                            x = max(0, min(x, w-1)); y = max(0, min(y, h-1))
                            ww = max(1, min(ww, w - x)); hh = max(1, min(hh, h - y))
                            faces.append((x, y, ww, hh))
                    return img, faces
            except Exception:
                pass

        # Haar fallback
        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
                for (x, y, ww, hh) in dets:
                    faces.append((int(x), int(y), int(ww), int(hh)))
                return img, faces
            except Exception:
                pass

        return img, faces
