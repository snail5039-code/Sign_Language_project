# train.py
# 목적:
#  - dataset_split/train, dataset_split/val 폴더에 있는 .npy를 읽어서
#  - 아주 간단한 딥러닝 모델로 "라벨(WORD1501 등)"을 맞추도록 학습한다.
#  - 결과로 model.pth(가중치), label_map.json(라벨<->숫자 매핑)을 저장한다.

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1) Dataset 클래스: 폴더에서 npy를 읽어오는 역할
# -----------------------------
class NpyDataset(Dataset):
    def __init__(self, root_dir: Path, label_to_idx: dict[str, int]):
        """
        root_dir 구조:
          root_dir/
            WORD1501/
              xxx.npy
              yyy.npy
            WORD1502/
              zzz.npy
        label_to_idx:
          {"WORD1501":0, "WORD1502":1, ...}
        """
        self.samples = []  # (npy_path, label_idx) 목록

        for label_name, label_idx in label_to_idx.items():
            label_folder = root_dir / label_name
            if not label_folder.exists():
                continue

            for npy_path in sorted(label_folder.glob("*.npy")):
                self.samples.append((npy_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        npy_path, y = self.samples[i]

        # npy 로드: shape = (30, 2, 21, 3)
        x = np.load(npy_path).astype(np.float32)
        # 손목(0번) 기준 상대좌표로 변환
        wrist = x[:, :, 0:1, :2]          # (30,2,1,2)
        x[..., :2] = x[..., :2] - wrist

        # 스케일링 (값 범위 줄이기)
        x[..., 0] /= 500.0
        x[..., 1] /= 500.0

        # --- normalize xy to 0~1 ---
        # xy = x[..., :2]
        # m = float(xy.max())
        # if m > 0:
        #     x[..., :2] = xy / m
        # --------------------------
        # torch tensor로 변환
        x = torch.from_numpy(x)  # float32 tensor
        y = torch.tensor(y, dtype=torch.long)

        return x, y


# -----------------------------
# 2) 아주 간단한 모델(첫 버전):
#    입력 (30,2,21,3)을 그냥 펼쳐서(Flatten) 분류
# -----------------------------
class SimpleClassifier(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        input_dim = 30 * 2 * 21 * 3  # (30,2,21,3)을 전부 펼치면 길이 3780

        self.net = nn.Sequential(
            nn.Flatten(),                 # (B,30,2,21,3) -> (B,3780)
            nn.Linear(input_dim, 256),    # 3780 -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)   # 256 -> 클래스 수
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 3) 폴더에서 라벨 목록 만들기
# -----------------------------
def build_label_map(train_dir: Path):
    # train_dir 안에 있는 폴더 이름들(WORD1501 등)을 라벨로 사용
    labels = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    label_to_idx = {name: i for i, name in enumerate(labels)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


# -----------------------------
# 4) 정확도 계산 함수
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)               # (B, num_classes)
        pred = logits.argmax(dim=1)     # 가장 큰 값의 인덱스 = 예측 라벨

        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / max(1, total)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_split", help="dataset_split 폴더 경로")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise RuntimeError(f"train 폴더 없음: {train_dir}")
    if not val_dir.exists():
        raise RuntimeError(f"val 폴더 없음: {val_dir}")

    # 라벨 맵 만들기
    label_to_idx, idx_to_label = build_label_map(train_dir)

    if len(label_to_idx) < 2:
        raise RuntimeError("라벨 폴더가 2개 이상 있어야 분류 학습이 됨 (WORD1501, WORD1502 등)")

    print("labels:", label_to_idx)

    # Dataset / DataLoader
    train_ds = NpyDataset(train_dir, label_to_idx)
    val_ds = NpyDataset(val_dir, label_to_idx)

    print("train y unique:", sorted(set([y for _, y in train_ds.samples])))
    print("train y head20:", [y for _, y in train_ds.samples[:20]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("train samples:", len(train_ds))
    print("val samples:  ", len(val_ds))

    # 디바이스 설정 (GPU 있으면 cuda, 없으면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 모델/손실/옵티마이저
    model = SimpleClassifier(num_classes=len(label_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # 학습 루프
    # -----------------------------
    printed = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if not printed:
                w = model.net[1].weight
                print("grad mean:", w.grad.abs().mean().item())
                printed = True

            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        avg_loss = total_loss / max(1, total)
        val_acc = evaluate(model, val_loader, device)

        print(f"epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    # -----------------------------
    # 모델/라벨맵 저장
    # -----------------------------
    out_model = Path("model.pth")
    out_map = Path("label_map.json")

    torch.save(model.state_dict(), out_model)
    out_map.write_text(json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_model)
    print("saved:", out_map)


if __name__ == "__main__":
    main()
