# Code/make_pupil_dataset_vr.py

from pathlib import Path
import pandas as pd

from loader import EyeDataset
from preprocess import Preprocessor
from pupil_segmentation import process_frame_vr


def build_pupil_csv_for_subject(subject: str, out_dir: Path):
    ds = EyeDataset()
    prep = Preprocessor(use_face_crop=False)  # VR에는 face crop 필요 X

    frames = ds.load_vr(subject, exclude_closed_eye=False)

    rows = []

    for f in frames:
        cx, cy, r, bin_img, vis = process_frame_vr(f, prep)

        # 실패 프레임은 0으로 두기 (나중에 필터링 가능하게)
        if cx is None:
            cx = cy = r = 0.0

        rows.append({
            "FILENAME": f.filename,
            "PRED_CENTER_X": cx,
            "PRED_CENTER_Y": cy,
            "PRED_RADIUS": r,
        })

        # 결과 시각화 이미지 저장
        import cv2
        out_img_dir = out_dir / "images" / subject
        out_img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_dir / f.filename), vis)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{subject}_VR_pupil_pred.csv"
    df.to_csv(csv_path, index=False)
    print(f"{subject}: saved {csv_path}")


if __name__ == "__main__":
    out_dir = Path("../results/pupil_dataset_vr")

    for sub in ["s01", "s02", "s03", "s04", "t01", "t02"]:
        build_pupil_csv_for_subject(sub, out_dir)
