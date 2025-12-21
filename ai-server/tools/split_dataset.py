# tools/split_dataset.py
# 목적:
#   “학습용/시험용 데이터로 정리하는 코드”
#   dataset_raw 폴더(라벨별로 npy가 모여있는 폴더)를
#   dataset_split/train 과 dataset_split/val 로 나누어 복사한다.
#
# 입력 폴더 예시(dataset_raw):
#   dataset_raw/
#     WORD1501/
#       a.npy
#       b.npy
#     WORD1502/
#       c.npy
#
# 출력 폴더 예시(dataset_split):
#   dataset_split/
#     train/
#       WORD1501/ (80%)
#       WORD1502/ (80%)
#     val/
#       WORD1501/ (20%)
#       WORD1502/ (20%)

import argparse   # 터미널 옵션(--in_dir 등) 받기
import random     # 파일 섞기(shuffle)용
import shutil     # 파일 복사(copy2)용
from pathlib import Path  # 경로(폴더/파일) 다루기 쉽게

def main():
    # -------------------------------
    # 1) 터미널 옵션(인자) 설정/읽기
    # -------------------------------
    ap = argparse.ArgumentParser()

    # --in_dir : 입력 폴더 경로 (dataset_raw)
    ap.add_argument("--in_dir", required=True, help="입력 폴더 경로 (예: .\\dataset_raw)")

    # --out_dir : 출력 폴더 경로 (dataset_split)
    ap.add_argument("--out_dir", required=True, help="출력 폴더 경로 (예: .\\dataset_split)")

    # --val_ratio : 전체에서 검증(val)로 보낼 비율 (기본 0.2 = 20%)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="val 비율 (기본 0.2)")

    # --seed : 섞는 결과를 항상 똑같이 만들기 위한 숫자
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본 42)")

    # 실제로 옵션들을 읽어서 args에 저장
    args = ap.parse_args()

    # -------------------------------
    # 2) 경로 준비 (문자열 -> Path 객체)
    # -------------------------------
    in_dir = Path(args.in_dir)    # 예: .\dataset_raw
    out_dir = Path(args.out_dir)  # 예: .\dataset_split

    # 출력 폴더 아래에 train/val 폴더 만들기
    train_root = out_dir / "train"
    val_root = out_dir / "val"

    # 폴더가 없으면 만들고, 이미 있으면 그냥 넘어감
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 3) 랜덤 섞기 고정 (재현성)
    # -------------------------------
    random.seed(args.seed)

    # -------------------------------
    # 4) 통계용 변수(몇 개 처리했는지)
    # -------------------------------
    labels = 0       # 라벨 폴더 개수 (WORD1501 같은 폴더)
    train_total = 0  # train에 복사된 파일 총 수
    val_total = 0    # val에 복사된 파일 총 수

    # -------------------------------
    # 5) 입력 폴더(dataset_raw) 안의 "라벨 폴더"들 반복
    #    예: dataset_raw/WORD1501, dataset_raw/WORD1502 ...
    # -------------------------------
    label_dirs = sorted([p for p in in_dir.iterdir() if p.is_dir()])

    for label_dir in label_dirs:
        # label_dir.name 은 "WORD1501" 같은 폴더 이름
        label_name = label_dir.name

        # 현재 라벨 폴더 안의 npy 파일들을 전부 찾기
        npy_files = sorted(label_dir.glob("*.npy"))

        # 파일이 없으면 스킵
        if not npy_files:
            continue

        labels += 1  # 라벨 하나 처리 시작

        # -------------------------------
        # 6) 파일 목록을 섞어서 랜덤 분리
        # -------------------------------
        random.shuffle(npy_files)

        # 전체 파일 개수
        n = len(npy_files)

        # val로 보낼 개수 = 전체의 val_ratio
        # 단, 최소 1개는 val로 보내도록 max(1, ...) 사용
        n_val = max(1, int(n * args.val_ratio))

        # 앞쪽 n_val개는 val, 나머지는 train
        val_files = npy_files[:n_val]
        train_files = npy_files[n_val:]

        # -------------------------------
        # 7) 출력 폴더 만들기
        #    dataset_split/train/WORD1501
        #    dataset_split/val/WORD1501
        # -------------------------------
        train_label_dir = train_root / label_name
        val_label_dir = val_root / label_name

        train_label_dir.mkdir(parents=True, exist_ok=True)
        val_label_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------
        # 8) 실제 파일 복사
        # -------------------------------
        for f in train_files:
            # f는 원본 파일 경로 (예: dataset_raw/WORD1501/a.npy)
            # 목적지 경로 (예: dataset_split/train/WORD1501/a.npy)
            dst = train_label_dir / f.name
            shutil.copy2(f, dst)

        for f in val_files:
            # 목적지 경로 (예: dataset_split/val/WORD1501/a.npy)
            dst = val_label_dir / f.name
            shutil.copy2(f, dst)

        # 통계 업데이트
        train_total += len(train_files)
        val_total += len(val_files)

    # -------------------------------
    # 9) 결과 출력
    # -------------------------------
    print(f"DONE labels={labels} train={train_total} val={val_total}")
    print("train_dir:", train_root)
    print("val_dir:  ", val_root)

# 파이썬 파일을 직접 실행하면 main()을 실행
if __name__ == "__main__":
    main()
