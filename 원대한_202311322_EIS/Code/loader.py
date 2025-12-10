# loader.py

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class EyeFrameSample:
    subject: int          # 피험자 번호 (1, 2, 3, ...)
    dist: int             # 30 or 50
    pos: str              # 'F', 'L', 'R'
    frame: int            # 프레임 번호 (0001, 0002, ...)
    rgb: np.ndarray       # (H, W, 3)
    ir: np.ndarray        # (H, W)
    depth: np.ndarray     # (H, W)
    gaze_xy: np.ndarray   # (N, 2) or (2,)  - csv 포맷에 따라

def make_filename(sub: int, dist: int, modality: str,
                  pos: str, frame: int, ext: str) -> str:

    return f"s{sub:02d}_{dist}_{modality}_{pos}_{frame:04d}{ext}"


def load_eye_frame(sub: int, dist: int, pos: str, frame: int) -> EyeFrameSample:

    cwd = Path.cwd()
    dataset_root = cwd.parent.parent / "dataset"

    # ex) dataset/s01/30/DEPTH
    subject_dir = dataset_root / f"s{sub:02d}"
    dist_dir    = subject_dir / str(dist)

    depth_dir = dist_dir / "DEPTH"
    ir_dir    = dist_dir / "IR"
    rgb_dir   = dist_dir / "RGB"
    xy_dir    = dist_dir / "XY"

    # 파일 이름
    depth_name = make_filename(sub, dist, "DEPTH", pos, frame, ".png")
    ir_name    = make_filename(sub, dist, "IR",    pos, frame, ".png")
    rgb_name   = make_filename(sub, dist, "RGB",   pos, frame, ".jpg")
    xy_name    = make_filename(sub, dist, "XY",    pos, frame, ".csv")

    # 전체 경로
    depth_path = depth_dir / depth_name
    ir_path    = ir_dir / ir_name
    rgb_path   = rgb_dir / rgb_name
    xy_path    = xy_dir / xy_name

    # 이미지 읽기
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    ir    = cv2.imread(str(ir_path),    cv2.IMREAD_GRAYSCALE)
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if depth is None:
        raise FileNotFoundError(f"Depth 이미지 로드 실패: {depth_path}")
    if ir is None:
        raise FileNotFoundError(f"IR 이미지 로드 실패: {ir_path}")
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB 이미지 로드 실패: {rgb_path}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    gaze_df = pd.read_csv(xy_path)
    gaze_xy = gaze_df[['x', 'y']].to_numpy()

    return EyeFrameSample(
        subject=sub,
        dist=dist,
        pos=pos,
        frame=frame,
        rgb=rgb,
        ir=ir,
        depth=depth,
        gaze_xy=gaze_xy
    )


if __name__ == "__main__":
    """
     # sample code
    sample = load_eye_frame(sub=1, dist=30, pos='F', frame=1)
    print("RGB shape :", sample.rgb.shape)
    print("IR shape  :", sample.ir.shape)
    print("Depth shape:", sample.depth.shape)
    print("Gaze head :", sample.gaze_xy[:5])"""
