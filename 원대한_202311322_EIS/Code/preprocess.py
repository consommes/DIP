# Code/preprocess.py

import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp


class Preprocessor:
    def __init__(self, use_face_crop=True):
        """
        use_face_crop=True → Type1에서 얼굴 crop 수행
        VR(Type2)에서는 crop 필요 없음
        """
        self.use_face_crop = use_face_crop

        # mediapipe 초기화
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    # ---------------------------------------------------------
    # 공통 유틸리티 함수
    # ---------------------------------------------------------
    def _read_gray(self, path: Path):
        """흑백 이미지 읽기 (segmentation에서 대부분 grayscale 사용)"""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _apply_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)

    def _gaussian(self, img, k=5):
        return cv2.GaussianBlur(img, (k, k), 0)

    # ---------------------------------------------------------
    # Type1: 1024×1024 얼굴 전체 영상 처리
    # ---------------------------------------------------------
    def preprocess_type1(self, frame):
        """
        1) grayscale load
        2) (optional) face crop
        3) gaussian blur
        4) CLAHE
        """
        img = self._read_gray(frame.path)

        if self.use_face_crop:
            img = self._face_crop(img)

        img = self._gaussian(img, k=5)
        img = self._apply_clahe(img)

        return img

    # ---------------------------------------------------------
    # Type2: VR 192×192 영상 → 매우 간단
    # ---------------------------------------------------------
    def preprocess_type2(self, frame):
        img = self._read_gray(frame.path)

        img = self._gaussian(img, k=3)
        img = self._apply_clahe(img)

        return img

    # ---------------------------------------------------------
    # 얼굴 영역 Crop (Type1 전용)
    # ---------------------------------------------------------
    def _face_crop(self, img):
        """
        mediapipe face detector를 이용해 bounding box crop
        """
        # mediapipe는 RGB 입력 요구 → 변환
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        results = self.mp_face.process(rgb)

        if not results.detections:
            # 얼굴을 못 찾으면 원본 그대로 반환
            return img

        det = results.detections[0]
        h, w = img.shape

        # bounding box 계산
        bbox = det.location_data.relative_bounding_box
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # crop
        x1 = max(0, xmin)
        y1 = max(0, ymin)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        cropped = img[y1:y2, x1:x2]

        # 혹시라도 비정상적으로 너무 작은 크기가 되면 원본 유지
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            return img

        return cropped
