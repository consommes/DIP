from loader import EyeDataset, DATASET_ROOT

print("DATASET_ROOT =", DATASET_ROOT)

ds = EyeDataset()   # 인자 없이 호출 → 자동으로 dataset 폴더 사용

# 1) Type1 예시: s01, 30cm, IR
frames_30_ir = ds.load_type1("s01", "30", "IR")
print("s01 30cm IR 프레임 수:", len(frames_30_ir))
print(frames_30_ir[0])

# 2) VR 예시
frames_vr = ds.load_vr("s01")
print("s01 VR 프레임 수:", len(frames_vr))
print(frames_vr[0])
