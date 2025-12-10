# pupil_segmentation.py

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from loader import EyeFrameSample, load_eye_frame


@dataclass
class PupilSegmentationResult:
    success: bool                  # 동공 검출 성공 여부
    center: Tuple[float, float]    # (cx, cy)
    axes: Tuple[float, float]      # (major, minor) or (a, b)
    angle: float                   # 타원 기울기 (deg)
    mask: np.ndarray               # 동공 binary mask (uint8, 0/255)
    contour: Optional[np.ndarray]  # 최종 contour (Nx1x2) 혹은 None


# -------------------------------
# 1) 전처리
# -------------------------------

def preprocess_ir(ir: np.ndarray) -> np.ndarray:
    """
    IR 영상을 pupil segmentation에 적합하게 전처리.
    입력 ir: (H, W), dtype: uint8 / uint16 / float 가능
    출력: uint8, (H, W)
    """

    # 1. 타입 정규화 → uint8 [0,255]
    if ir.dtype != np.uint8:
        ir_norm = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX)
        ir_u8 = ir_norm.astype(np.uint8)
    else:
        ir_u8 = ir.copy()

    # 2. 히스토그램 평활화로 대비 향상
    ir_eq = cv2.equalizeHist(ir_u8)

    # 3. Gaussian blur (노이즈 감소, 너무 세게는 X)
    ir_blur = cv2.GaussianBlur(ir_eq, (5, 5), 0)

    return ir_blur


# -------------------------------
# 2) pupil 후보 이진화 & 후처리
# -------------------------------

def binarize_pupil(ir_preprocessed: np.ndarray) -> np.ndarray:
    """
    동공(어두운 부분)을 흰색(255)으로 만드는 binary mask 생성
    """

    # Otsu + inverse: dark가 foreground가 되도록
    _, th = cv2.threshold(
        ir_preprocessed,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 형태학적 closing으로 구멍 메우기 & 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 너무 작은 잡음 날리기 (open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


# -------------------------------
# 3) contour 기반 ellipse fitting
# -------------------------------

def find_pupil_ellipse(mask: np.ndarray) -> PupilSegmentationResult:
    """
    binary mask에서 contour를 찾고,
    그 중 pupil로 가장 그럴듯한 contour에 대해 ellipse fitting 수행.
    """

    h, w = mask.shape[:2]
    img_center = np.array([w / 2.0, h / 2.0])

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return PupilSegmentationResult(
            success=False,
            center=(0.0, 0.0),
            axes=(0.0, 0.0),
            angle=0.0,
            mask=mask,
            contour=None
        )

    best_score = -1.0
    best_ellipse = None
    best_contour = None

    # 면적 기준 (영상 크기에 따라 대략적인 비율로 설정)
    img_area = float(w * h)
    min_area = img_area * 0.001   # 너무 작으면 noise
    max_area = img_area * 0.2     # 너무 크면 얼굴 전체 등

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        if len(cnt) < 5:
            # fitEllipse는 최소 5점 필요
            continue

        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (major, minor), angle = ellipse

        # 1) 중심이 영상 중앙과 얼마나 가까운지
        center_dist = np.linalg.norm(np.array([cx, cy]) - img_center)

        # 2) circularity: 4*pi*A / P^2
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)

        # circularity가 0.4 ~ 1.2 사이 정도를 기대 (정확한 값은 데이터 보며 조정)
        if not (0.3 <= circularity <= 1.3):
            continue

        # score: circularity에 가중치, 중심이 가까울수록 좋게
        # (점수 정의는 자유)
        score = circularity - 0.001 * center_dist

        if score > best_score:
            best_score = score
            best_ellipse = ellipse
            best_contour = cnt

    if best_ellipse is None:
        # 적합 candidate 없음
        return PupilSegmentationResult(
            success=False,
            center=(0.0, 0.0),
            axes=(0.0, 0.0),
            angle=0.0,
            mask=mask,
            contour=None
        )

    (cx, cy), (major, minor), angle = best_ellipse

    return PupilSegmentationResult(
        success=True,
        center=(float(cx), float(cy)),
        axes=(float(major / 2.0), float(minor / 2.0)),  # 반지름 형태로 저장
        angle=float(angle),
        mask=mask,
        contour=best_contour
    )


# -------------------------------
# 4) 전체 파이프라인 함수
# -------------------------------

def segment_pupil_from_sample(sample: EyeFrameSample) -> PupilSegmentationResult:
    """
    EyeFrameSample 하나(IR)에서 pupil segmentation 수행.
    """
    ir = sample.ir
    ir_pre = preprocess_ir(ir)
    mask = binarize_pupil(ir_pre)
    result = find_pupil_ellipse(mask)
    return result


def visualize_result(sample: EyeFrameSample,
                     result: PupilSegmentationResult) -> np.ndarray:
    """
    IR 영상 위에 동공 타원을 시각화한 RGB 이미지 반환.
    """
    ir = sample.ir

    # 시각화를 위해 grayscale → BGR
    if ir.dtype != np.uint8:
        ir_norm = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX)
        ir_u8 = ir_norm.astype(np.uint8)
    else:
        ir_u8 = ir.copy()

    vis_bgr = cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR)

    if result.success and result.contour is not None:
        # 타원 그리기
        center = (int(result.center[0]), int(result.center[1]))
        axes = (int(result.axes[0]), int(result.axes[1]))
        angle = result.angle

        cv2.ellipse(
            vis_bgr,
            center=center,
            axes=axes,
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=(0, 255, 255),   # 노랑
            thickness=2
        )

        # center point
        cv2.circle(
            vis_bgr,
            center,
            radius=3,
            color=(0, 0, 255),     # 빨강
            thickness=-1
        )

    return vis_bgr


# -------------------------------
# 5) 간단 테스트
# -------------------------------

if __name__ == "__main__":
    # 예: s01, 30cm, Front, frame 1
    sample = load_eye_frame(sub=1, dist=30, pos='F', frame=1)

    result = segment_pupil_from_sample(sample)
    print("Success:", result.success)
    print("Center :", result.center)
    print("Axes   :", result.axes)
    print("Angle  :", result.angle)

    vis = visualize_result(sample, result)
    cv2.imshow("IR + pupil", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
