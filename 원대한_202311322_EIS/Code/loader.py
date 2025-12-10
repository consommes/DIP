from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # 원대한_202311322_EIS/
DATASET_ROOT = PROJECT_ROOT / "dataset"      # .../dataset
LABEL_ROOT = DATASET_ROOT / "label"          # .../dataset/label


@dataclass
class EyeFrame:
    subject: str
    data_type: str        # "30", "50", "VR"
    modality: str         # "DEPTH", "IR", "RGB"
    filename: str         # 예: s01_30_IR_F_0001.png
    path: Path
    leye_x: float
    leye_y: float
    reye_x: float
    reye_y: float


class EyeDataset:
    def __init__(self, dataset_root: Path | str | None = None):
        if dataset_root is None:
            self.root = DATASET_ROOT
        else:
            self.root = Path(dataset_root).resolve()

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if not LABEL_ROOT.exists():
            raise FileNotFoundError(f"Label folder not found: {LABEL_ROOT}")

    # -------------------------------------------------
    # 1) Type1용 CSV: dataset/label/s01_30_DEPTH.csv ...
    # -------------------------------------------------
    def _load_label_csv(self, subject: str, data_type: str, modality: str) -> pd.DataFrame:
        csv_name = f"{subject}_{data_type}_{modality}.csv"
        csv_path = LABEL_ROOT / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        return df

    # -------------------------------------------------
    # 2) Type1 프레임 리스트
    # -------------------------------------------------
    def load_type1(
        self,
        subject: str,
        data_type: str,           # "30" or "50"
        modality: str,            # "DEPTH", "IR", "RGB"
        exclude_closed_eye: bool = True,
    ) -> List[EyeFrame]:

        df = self._load_label_csv(subject, data_type, modality)

        img_dir = self.root / subject / data_type / modality
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)

        frames: List[EyeFrame] = []

        for _, row in df.iterrows():
            filename = str(row["FILENAME"])
            path = img_dir / filename

            le_x = float(row["LEYE_CENTER_X"])
            le_y = float(row["LEYE_CENTER_Y"])
            re_x = float(row["REYE_CENTER_X"])
            re_y = float(row["REYE_CENTER_Y"])

            # 네 좌표가 다 0이면 눈을 감아서 center 못 잡는 프레임
            if exclude_closed_eye and le_x == 0 and le_y == 0 and re_x == 0 and re_y == 0:
                continue

            if not path.exists():
                continue

            frames.append(
                EyeFrame(
                    subject=subject,
                    data_type=data_type,
                    modality=modality,
                    filename=filename,
                    path=path,
                    leye_x=le_x,
                    leye_y=le_y,
                    reye_x=re_x,
                    reye_y=re_y,
                )
            )

        return frames

    # -------------------------------------------------
    # 3) VR용 CSV: dataset/label/s01_VR_IR.csv
    # -------------------------------------------------
    def _load_vr_csv(self, subject: str) -> pd.DataFrame:
        csv_name = f"{subject}_VR_IR.csv"
        csv_path = LABEL_ROOT / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        return pd.read_csv(csv_path)

    def load_vr(
        self,
        subject: str,
        exclude_closed_eye: bool = True,
    ) -> List[EyeFrame]:

        data_type = "VR"
        modality = "IR"

        df = self._load_vr_csv(subject)
        img_dir = self.root / subject / data_type / modality
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)

        frames: List[EyeFrame] = []

        for _, row in df.iterrows():
            filename = str(row["FILENAME"])
            path = img_dir / filename

            le_x = float(row["LEYE_CENTER_X"])
            le_y = float(row["LEYE_CENTER_Y"])
            re_x = float(row["REYE_CENTER_X"])
            re_y = float(row["REYE_CENTER_Y"])

            # VR에서도 네 좌표 모두 0이면 center 추출 불가 프레임
            if exclude_closed_eye and le_x == 0 and le_y == 0 and re_x == 0 and re_y == 0:
                continue

            if not path.exists():
                continue

            frames.append(
                EyeFrame(
                    subject=subject,
                    data_type=data_type,
                    modality=modality,
                    filename=filename,
                    path=path,
                    leye_x=le_x,
                    leye_y=le_y,
                    reye_x=re_x,
                    reye_y=re_y,
                )
            )

        return frames
