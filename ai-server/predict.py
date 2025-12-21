"""
predict.py
-------------------------
목적:
- 학습된 모델(model.pth) + 라벨맵(label_map.json)을 불러와서
- npy 파일 1개를 넣으면
- 모델이 예측한 WORD 라벨 Top-K를 출력한다.

왜 필요?
- 학습이 끝났다고 해서 바로 연결(서버/프론트)하면,
  "모델이 문제인지 / 연결이 문제인지" 구분이 안 됨.
- 그래서 모델만 단독으로 정상 예측하는지 먼저 확인하는 '시험기' 역할.

실행 예시:
python ./predict.py --npy "./dataset_split_small/val/WORD0001/NIA_SL_WORD0001_REAL17_D.npy"

(주의)
- 학습할 때 쓴 전처리(손목 기준 상대좌표 + 스케일링)를
  예측할 때도 똑같이 해야 함.
"""

import argparse      # 터미널에서 옵션(--npy, --model 등) 받는 용도
import json          # label_map.json 읽을 때 사용
from pathlib import Path  # 경로 문자열을 편하게 다루기 위한 라이브러리

import numpy as np   # npy 로드/전처리
import torch         # PyTorch
import torch.nn as nn  # 신경망 레이어들


# -------------------------
# 1) 학습 때 쓰던 모델 구조를 "그대로" 다시 적어야 함
# -------------------------
# train.py에서 쓴 SimpleClassifier와 같은 구조여야
# model.pth(가중치)를 로딩할 수 있음.
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # nn.Sequential = 레이어를 순서대로 이어 붙이는 방식
        self.net = nn.Sequential(
            nn.Flatten(),                   # (30,2,21,3) -> (30*2*21*3) 한 줄로 펼치기
            nn.Linear(30 * 2 * 21 * 3, 256), # 입력 -> 은닉층 256개
            nn.ReLU(),                      # 비선형 활성화 함수
            nn.Dropout(0.2),                # 과적합 방지용(학습 때랑 동일하게)
            nn.Linear(256, num_classes),    # 256 -> 클래스 수(예: WORD0001~ 등)
        )

    def forward(self, x):
        # forward = 모델이 입력을 받았을 때 출력(logits)을 내는 함수
        return self.net(x)


# -------------------------
# 2) label_map.json 로드
# -------------------------
# label_map.json 예시:
# {"WORD0001": 0, "WORD0002": 1, ...}
# 모델은 숫자(0,1,2...)로 예측하니까,
# 다시 WORD 라벨로 바꾸기 위해 역매핑(idx_to_label)을 만든다.
def load_label_map(label_map_path: Path):
    """
    label_map.json이 어떤 형태든 처리:
    1) {"WORD0001": 0, "WORD0002": 1, ...}  (WORD->IDX)
    2) {"0": "WORD0001", "1": "WORD0002", ...}  (IDX->WORD)  ← 너 케이스
    """
    d = json.loads(label_map_path.read_text(encoding="utf-8"))

    # 케이스 판별:
    # 키가 "0","1" 이런 숫자 문자열이면 -> IDX->WORD 형태
    # 키가 "WORD..."면 -> WORD->IDX 형태
    sample_key = next(iter(d.keys()))

    # 2) IDX->WORD
    if str(sample_key).isdigit():
        idx_to_label = {int(k): v for k, v in d.items()}
        label_to_idx = {v: int(k) for k, v in d.items()}
        return label_to_idx, idx_to_label

    # 1) WORD->IDX
    else:
        label_to_idx = {k: int(v) for k, v in d.items()}
        idx_to_label = {int(v): k for k, v in d.items()}
        return label_to_idx, idx_to_label


# -------------------------
# 3) 전처리: 학습 때 했던 것과 "똑같이" 해야 함
# -------------------------
# 너가 학습 성공시킨 핵심:
# - 손목(0번 포인트)을 기준점으로 잡아서 "상대좌표"로 바꿈
# - 값을 너무 크지 않게 /500 으로 스케일링
def preprocess_like_train(x: np.ndarray) -> np.ndarray:
    """
    입력 x는 npy를 로드한 배열
    shape: (30, 2, 21, 3)
      - 30프레임
      - 왼손/오른손 2개
      - 랜드마크 21개
      - (x, y, score/기타) 3차원
    """

    # float32로 변환 (딥러닝은 보통 float32 사용)
    x = x.astype(np.float32)

    # 손목(0번 랜드마크) 좌표만 뽑기
    # x[:, :, 0:1, :2]
    #  - : 30프레임 전체
    #  - : 왼/오손 전체
    #  - 0:1 0번 점(손목)만 (차원 유지 위해 0:1로 슬라이스)
    #  - :2 x,y만 사용
    wrist = x[:, :, 0:1, :2]   # shape: (30, 2, 1, 2)

    # 상대좌표로 만들기: 모든 점에서 손목 좌표를 빼준다
    # 이렇게 하면 "사람이 화면 어디에 있든" 손목이 기준점이 되어서 학습이 쉬워짐
    x[..., :2] = x[..., :2] - wrist

    # 값 스케일링 (너가 학습 때 사용한 값과 동일해야 함)
    x[..., 0] /= 500.0
    x[..., 1] /= 500.0

    return x


# -------------------------
# 4) 메인 실행 함수
# -------------------------
# @torch.no_grad() = 여기 안에서는 "학습"이 아니라 "예측"만 한다는 의미
# (그래서 메모리도 적게 쓰고 더 빠름)
@torch.no_grad()
def main():
    # -------------------------
    # 4-1) 터미널 옵션 받기
    # -------------------------
    ap = argparse.ArgumentParser()

    # --npy: 예측할 파일(필수)
    ap.add_argument("--npy", required=True, help="예측할 .npy 파일 경로")

    # --model: 학습된 가중치 파일(model.pth). 기본은 현재 폴더의 model.pth
    ap.add_argument("--model", default="model.pth", help="학습된 model.pth 경로")

    # --label_map: 라벨맵 파일(label_map.json). 기본은 현재 폴더의 label_map.json
    ap.add_argument("--label_map", default="label_map.json", help="label_map.json 경로")

    # --topk: 상위 몇 개 결과를 볼지 (기본 5개)
    ap.add_argument("--topk", type=int, default=5, help="top-k 출력 개수")

    args = ap.parse_args()

    # -------------------------
    # 4-2) 디바이스 선택 (GPU 있으면 cuda, 없으면 cpu)
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # -------------------------
    # 4-3) 경로 준비 및 존재 확인
    # -------------------------
    npy_path = Path(args.npy)
    model_path = Path(args.model)
    label_map_path = Path(args.label_map)

    if not npy_path.exists():
        raise FileNotFoundError(f"npy not found: {npy_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")

    # -------------------------
    # 4-4) 라벨맵 로드 (클래스 수 결정)
    # -------------------------
    label_to_idx, idx_to_label = load_label_map(label_map_path)
    num_classes = len(label_to_idx)

    # -------------------------
    # 4-5) 모델 만들고, 가중치(model.pth) 로드
    # -------------------------
    model = SimpleClassifier(num_classes=num_classes).to(device)

    # torch.load로 저장된 가중치를 읽고
    state = torch.load(model_path, map_location=device)

    # 모델에 가중치를 넣는다
    model.load_state_dict(state)

    # eval 모드 = 드롭아웃/배치정규화 등이 "예측용"으로 동작
    model.eval()

    # -------------------------
    # 4-6) npy 로드 -> 전처리 -> 텐서 변환
    # -------------------------
    x = np.load(npy_path)  # shape (30,2,21,3)
    x = preprocess_like_train(x)

    # torch 텐서로 변환
    x = torch.from_numpy(x)

    # 모델은 배치 차원이 필요하니까 맨 앞에 batch=1 차원을 추가
    # (30,2,21,3) -> (1,30,2,21,3)
    x = x.unsqueeze(0).to(device)

    # -------------------------
    # 4-7) 모델 예측
    # -------------------------
    logits = model(x)  # shape: (1, num_classes)

    # softmax = 점수를 확률처럼 바꿔주는 함수
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # shape: (num_classes,)

    # -------------------------
    # 4-8) Top-K 뽑기
    # -------------------------
    topk = min(args.topk, num_classes)

    # 확률이 큰 순서대로 인덱스 정렬
    top_idx = np.argsort(-probs)[:topk]

    # 결과 출력
    print(f"\nfile: {npy_path}")
    for rank, idx in enumerate(top_idx, 1):
        label = idx_to_label.get(int(idx), f"IDX_{idx}")
        print(f"{rank}) {label}  prob={probs[idx]:.4f}")


# python predict.py로 직접 실행되면 main()을 실행
if __name__ == "__main__":
    main()
