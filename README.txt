Task 1 – Eye Image Segmentation 과제 설명

과제 제출 기한 : 2025년 12월 12일 (금) – 15주차 금요일

[과제 목표]
본 과제의 목표는 눈 이미지에서 "Eye region"과 "Pupil"을 인식하는 것입니다.
각 프레임에서 눈의 중심점(center) 좌표를 찾고, 동공 영역을 원(circle) 형태로 표시하는 것이 최종 목표입니다.
(자세한 내용은 과제 설명 자료 참고)

[기본 규칙]
※ 수업에서 배운 기본 영상처리 기법만을 사용해야 합니다.
※ 딥러닝 / 머신러닝 기반 feature extraction 또는 classification은 금지합니다.

[제공 파일 설명]
- label 폴더에는 각 타입(Type 1/Type 2) 영상의 modality별로 REYE(오른쪽 눈)와 LEYE(왼쪽 눈) 중심 좌표(X, Y)가 포함된 CSV 파일을 제공합니다.
- s01, s02, s03, s04는 예시(example) 데이터로 label을 함께 제공합니다.
- t01, t02는 테스트(test) 데이터이며 label은 제공되지 않습니다.
- 파일 저장 방식 (sub: 피험자 번호 / dist: 거리 / pos: 자세 / frame: 프레임 번호):
    Depth : s{sub}_{dist}_DEPTH_{pos}_{frame}.png
    IR    : s{sub}_{dist}_IR_{pos}_{frame}.png
    RGB   : s{sub}_{dist}_RGB_{pos}_{frame}.jpg
    XY    : s{sub}_{dist}_XY_{pos}_{frame}.csv

[Dataset 디렉토리 구조]
dataset/
 ├─ label/
 │    ├─ s01_30_DEPTH.csv
 │    ├─ s01_30_IR.csv
 │    ├─ s01_30_RGB.csv
 │    └─ ...
 ├─ s01/
 │    ├─ 30/
 │    │    ├─ DEPTH
 │    │    ├─ IR
 │    │    ├─ RGB
 │    │    └─ XY
 │    ├─ 50/
 │    │    ├─ DEPTH
 │    │    ├─ IR
 │    │    ├─ RGB
 │    │    └─ XY
 │    └─ VR/
 │         ├─ IR
 │         └─ XY
 ├─ s02/
 ├─ s03/
 ├─ s04/
 ├─ t01/
 └─ t02/

[제출 파일]
1. Source Code
   - 구현 언어: Python (기본)
2. Report
   - Eye region & Pupil detection 과정에서 사용한 영상처리 기법 설명
   - Example 데이터(s01–s04) 중 프레임 일부를 사용해 결과 시각화 포함
3. Results
   - Test set(t01, t02)에 대한 predict.csv 제출

제출 파일명:
   ‘이름_학번_EIS.zip’   (EIS = Eye Image Segmentation)

※ 제공된 디렉토리의 파일들은 템플릿이므로, 이름/학번을 채우고 본인 결과물을 채워 넣으면 됩니다.
※ 용량이 너무 커서 e-campus 업로드 오류가 발생할 경우, 워드/한글 문서 대신 PDF로 변환해서 제출해주시면 됩니다.