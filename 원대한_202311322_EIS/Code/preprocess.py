# Code/preprocess.py

import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp  # pip install mediapipe 필요


class Preprocessor:
    def __init__(self, use_face_crop: bool = True):
        """
        use_face_crop : Type1(30/50) 이미지에서 얼굴만 잘라낼지 여부
        VR(Type2)에는 영향 없음.
        """
        self.use_face_crop = use_face_crop

        # mediapipe face detector 초기화
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    # -------------------------------------------------
    # 공통: 흑백 이미지 로드 + 간단 유틸
    # -------------------------------------------------
    def _read_gray(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _gaussian(self, img: np.ndarray, k: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(img, (k, k), 0)

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    # -------------------------------------------------
    # 얼굴 영역 crop (Type1만 사용)
    # -------------------------------------------------
    def _face_crop(self, img: np.ndarray) -> np.ndarray:
        """
        mediapipe로 얼굴 bounding box 찾아서 crop.
        얼굴 못 찾으면 원본 그대로 반환.
        """
        h, w = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        results = self.mp_face.process(rgb)

        if not results.detections:
            return img  # 얼굴 못 찾으면 그대로

        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        cropped = img[y1:y2, x1:x2]

        # 혹시 너무 작게 잘리면 (검출이 이상하면) 그냥 원본 사용
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            return img

        return cropped

    # -------------------------------------------------
    # Type1 (30/50) 전처리
    # -------------------------------------------------
    def preprocess_type1(self, frame) -> np.ndarray:
        """
        EyeFrame 하나를 받아서 전처리된 그레이 이미지를 리턴.
        - grayscale load
        - (옵션) face crop
        - Gaussian blur
        - CLAHE
        """
        img = self._read_gray(frame.path)

        if self.use_face_crop:
            img = self._face_crop(img)

        img = self._gaussian(img, k=5)
        img = self._clahe(img)

        return img

    # -------------------------------------------------
    # Type2 (VR) 전처리
    # -------------------------------------------------
    def preprocess_vr(self, frame) -> np.ndarray:
        """
        VR IR 영상 전처리:
        - grayscale load
        - Gaussian blur
        - CLAHE
        """
        img = self._read_gray(frame.path)
        img = self._gaussian(img, k=3)
        img = self._clahe(img)
        return img
