from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List
from predict import SimpleClassifier, load_label_map, preprocess_like_train
from pathlib import Path
import numpy as np
import torch
import json

app = FastAPI()

device = "cpu"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "current"   # current 폴더에서만 모델 읽기

T = 30
HAND_POINTS = 21
FACE_POINTS = 478


def load_label_to_text():
    path = MODEL_DIR / "label_to_text.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


label_to_text = load_label_to_text()

label_to_idx, idx_to_label = load_label_map(MODEL_DIR / "label_map.json")
model = SimpleClassifier(num_classes=len(label_to_idx)).to(device)

state = torch.load(MODEL_DIR / "model.pth", map_location=device)
model.load_state_dict(state)
model.eval()


class TranslateRequest(BaseModel):
    frames: List[Any]

    framesReceivedAny = len(seq_hand)

    # ✅ 손 프레임이 0개면 번역 안함
    if framesReceivedAny == 0:
        return {
            "text": "번역 실패",
            "confidence": 0.0,
            "framesReceived": 0,
            "framesReceivedHand": 0,
            "framesReceivedFace": 0,
            "framesReceivedAny": 0,
        }

    # 길이 T(30) 맞추기
    if framesReceivedAny >= T:
        seq_hand = seq_hand[-T:]
        seq_face = seq_face[-T:]
    else:
        pad_count = T - framesReceivedAny
        seq_hand += [np.zeros((2, HAND_POINTS, 3), dtype=np.float32) for _ in range(pad_count)]
        seq_face += [np.zeros((FACE_POINTS, 3), dtype=np.float32) for _ in range(pad_count)]

    x_hand = np.stack(seq_hand, axis=0)  # (30,2,21,3)

    # ✅ 지금 모델이 손 입력만 받는 구조라서(너 predict.py 기준) 손만 전처리/예측
    x2 = preprocess_like_train(x_hand)
    xt = torch.from_numpy(x2).unsqueeze(0).to(device)  # (1,30,2,21,3)

    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    label = idx_to_label[pred_idx]
    human_text = label_to_text.get(label, label)
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "text": human_text,
        "confidence": confidence,
        "framesReceived": framesReceivedHand,
        "framesReceivedHand": framesReceivedHand,
        "framesReceivedFace": framesReceivedFace,
        "framesReceivedAny": framesReceivedAny,
    }


class LabelMapItem(BaseModel):
    label: str
    text: str
    def make_seq(has_hand: bool, has_face: bool, n=T):
        return [make_frame(has_hand, has_face) for _ in range(n)]

    def run_case(title, frames):
        print("\n==============================")
        print(title)
        r = client.post("/predict", json={"frames": frames})
        print("status:", r.status_code)
        try:
            print("json:", r.json())
        except Exception:
            print("text:", r.text)