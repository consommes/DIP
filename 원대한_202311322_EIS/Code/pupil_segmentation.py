# Code/pupil_segmentation.py

import cv2
import numpy as np
from typing import Tuple, Optional

from loader import EyeFrame
from preprocess import Preprocessor


# ---------------------------------------------------------
# 공통: pupil 시각화
# ---------------------------------------------------------
def draw_pupil_on_image(
    img: np.ndarray,
    cx: Optional[float],
    cy: Optional[float],
    r: Optional[float],
    color=(0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    BGR 이미지에 pupil center와 원을 그려서 반환.
    """
    out = img.copy()

    if cx is None or cy is None or r is None:
        return out

    center = (int(round(cx)), int(round(cy)))
    radius = int(round(r))

    cv2.circle(out, center, radius, color, thickness)
    cv2.circle(out, center, 2, (0, 255, 0), -1)  # 중심점(초록색)

    return out


# ---------------------------------------------------------
# 1) VR (Type2) 전용 pupil segmentation - 최종 설계안
# ---------------------------------------------------------
def segment_pupil_vr_refined(
    gray: np.ndarray,
    crop_margin: int = 40,
    min_area: int = 40,
    max_area_ratio: float = 0.25,
    alpha_center: float = 40.0,
    min_radius: float = 5.0,
    max_radius: float = 40.0,
) -> Tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
    """
    VR eye (Type2) 전용 pupil segmentation.

    알고리즘:
      1) 중앙 영역만 crop (eyelid / 모서리 그림자 제거)
      2) CLAHE + Gaussian blur
      3) adaptive threshold (mean, inverse)
      4) morphology (open -> close)
      5) connected components 분석으로 pupil blob 선택
      6) minEnclosingCircle으로 center / radius 추정

    반환: (cx, cy, r, bin_full)
        - cx, cy, r : 원본 이미지 좌표계(전체 gray 기준)
        - bin_full  : 시각화용 이진 이미지 (원본 크기)
    """
    H, W = gray.shape

    # 1) 중앙 crop (대략 40~(H-40) 구간)
    y1 = crop_margin
    y2 = H - crop_margin
    x1 = crop_margin
    x2 = W - crop_margin

    if y2 <= y1 or x2 <= x1:
        # margin이 이미지보다 크면 그냥 전체 사용
        y1, x1, y2, x2 = 0, 0, H, W

    crop = gray[y1:y2, x1:x2]
    ch, cw = crop.shape
    crop_center = (cw / 2.0, ch / 2.0)

    # 2) CLAHE + blur (로컬 대비 강화 후 부드럽게)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop_enh = clahe.apply(crop)
    crop_blur = cv2.GaussianBlur(crop_enh, (5, 5), 0)

    # 3) adaptive threshold (pupil = 흰색)
    bin_crop = cv2.adaptiveThreshold(
        crop_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=8,
    )

    # 4) morphology (작은 잡음 제거 + 구멍 메우기)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_crop = cv2.morphologyEx(bin_crop, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_crop = cv2.morphologyEx(bin_crop, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5) connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_crop)

    best_label = -1
    best_score = 1e12
    crop_area = ch * cw
    max_area = max_area_ratio * crop_area

    for label in range(1, num_labels):  # 0 = background
        x, y, bw, bh, area = stats[label]

        # (1) 면적 필터
        if area < min_area:
            continue
        if area > max_area:
            continue

        # (2) bounding box가 crop 바운더리에 붙어 있으면 제외 (eyelid / 모서리 그림자)
        if x == 0 or y == 0 or (x + bw) >= cw or (y + bh) >= ch:
            continue

        # (3) aspect ratio (너무 납작한 blob 제외)
        aspect = bw / float(bh) if bh > 0 else 1.0
        if aspect < 0.4 or aspect > 2.5:
            continue

        # (4) blob mask
        mask = (labels == label).astype(np.uint8) * 255

        # (5) 원래 그레이에서 평균 밝기 (더 어두운 blob 선호)
        mean_int = cv2.mean(crop, mask=mask)[0]

        # (6) 중심과 crop 중앙 거리 (중앙에 가까울수록 좋음)
        cx_c, cy_c = centroids[label]
        dx = (cx_c - crop_center[0]) / (cw / 2.0)
        dy = (cy_c - crop_center[1]) / (ch / 2.0)
        center_dist = np.sqrt(dx * dx + dy * dy)  # 0~대략 1

        # (7) 근사 반지름 범위 체크
        r_est = np.sqrt(area / np.pi)
        if not (min_radius <= r_est <= max_radius):
            continue

        # (8) 점수 (낮을수록 좋음)
        score = mean_int + alpha_center * center_dist

        if score < best_score:
            best_score = score
            best_label = label

    if best_label == -1:
        # 실패 시: bin_full만 채워서 반환
        bin_full = np.zeros_like(gray)
        bin_full[y1:y2, x1:x2] = bin_crop
        return None, None, None, bin_full

    # 선택된 blob의 픽셀들로부터 최소 포함 원
    ys, xs = np.where(labels == best_label)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2) in crop coords
    (cx_crop, cy_crop), radius = cv2.minEnclosingCircle(pts)

    # crop 좌표 → 원본 이미지 좌표
    cx = x1 + cx_crop
    cy = y1 + cy_crop

    # bin_full에 crop 결과 삽입
    bin_full = np.zeros_like(gray)
    bin_full[y1:y2, x1:x2] = bin_crop

    return float(cx), float(cy), float(radius), bin_full


def process_frame_vr(
    frame: EyeFrame,
    preprocessor: Preprocessor,
):
    """
    Type2(VR) 프레임 처리:
      - preprocess_vr() → segment_pupil_vr_refined() → overlay 이미지
    """
    gray = preprocessor.preprocess_vr(frame)

    cx, cy, r, bin_img = segment_pupil_vr_refined(gray)
    vis_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = draw_pupil_on_image(vis_gray_bgr, cx, cy, r)

    return cx, cy, r, bin_img, overlay


# ---------------------------------------------------------
# 2) Type1용 (모니터 30cm / 50cm) – 원래 기본 버전 유지
# ---------------------------------------------------------
def segment_pupil_basic(
    gray: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
    """
    Type1 케이스에서 쓸 수 있는 기본 pupil segmentation.
    (global Otsu + morphology + contour + minEnclosingCircle)
    """
    H, W = gray.shape

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bin_img = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None, bin_img

    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue

        circularity = 4 * np.pi * area / (perim * perim)
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        center_penalty = (
            abs(x - W / 2) / (W / 2) +
            abs(y - H / 2) / (H / 2)
        )

        score = circularity - 0.3 * center_penalty

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None, None, None, bin_img

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    return float(cx), float(cy), float(radius), bin_img


def process_frame_type1(
    frame: EyeFrame,
    preprocessor: Preprocessor,
):
    gray = preprocessor.preprocess_type1(frame)
    cx, cy, r, bin_img = segment_pupil_basic(gray)
    vis_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = draw_pupil_on_image(vis_gray_bgr, cx, cy, r)
    return cx, cy, r, bin_img, overlay
