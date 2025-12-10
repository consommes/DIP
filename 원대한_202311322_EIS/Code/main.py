# Code/main.py

from pathlib import Path
import cv2

from loader import EyeDataset
from preprocess import Preprocessor
from pupil_segmentation import process_frame_type1, process_frame_vr


def test_segmentation():
    ds = EyeDataset()
    prep = Preprocessor(use_face_crop=True)

    out_dir = Path("../results/segmentation_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Type1 예시: s01 / 30 / IR
    frames_30_ir = ds.load_type1("s01", "30", "IR")
    f = frames_30_ir[0]

    cx, cy, r, bin_img, overlay = process_frame_type1(f, prep)

    cv2.imwrite(str(out_dir / "s01_30_IR_binary.png"), bin_img)
    cv2.imwrite(str(out_dir / "s01_30_IR_overlay.png"), overlay)

    print("Type1: cx, cy, r =", cx, cy, r)

    # 2) VR 예시
    frames_vr = ds.load_vr("s01")
    f_vr = frames_vr[0]

    cx2, cy2, r2, bin_img2, overlay2 = process_frame_vr(f_vr, prep)

    cv2.imwrite(str(out_dir / "s01_VR_IR_binary.png"), bin_img2)
    cv2.imwrite(str(out_dir / "s01_VR_IR_overlay.png"), overlay2)

    print("VR: cx, cy, r =", cx2, cy2, r2)

    print("결과 저장 경로:", out_dir)


if __name__ == "__main__":
    test_segmentation()
