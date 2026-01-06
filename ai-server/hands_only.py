# ai_server/hands_only.py
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter
from pydantic import BaseModel, Field

# ====== 설정 ======
T = int(os.getenv("T", "30"))
HAND_FEAT = 126

DEBUG = os.getenv("DEBUG", "0") == "1"
BASE_DIR = Path(__file__).resolve().parent
CURRENT = BASE_DIR / "current"
LABEL_MAP_PATH = CURRENT / "label_map.json"
LABEL_TO_TEXT_PATH = CURRENT / "label_to_text.json"
MODEL_PATH = CURRENT / "best_model.pth"

THRESHOLD = float(os.getenv("THRESHOLD", "0.60"))
STREAK_N = int(os.getenv("STREAK_N", "8"))

# 모델 입력 차원(기존 336 모델이면 336, 손만 새학습이면 126)
MODEL_FEAT_DIM = int(os.getenv("MODEL_FEAT_DIM", "336"))

# ====== 모델 ======
class TCNClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        h = self.backbone(x)      # (B,128,T)
        h = h.mean(dim=2)         # (B,128)
        return self.head(h)       # (B,C)

def load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

idx_to_label = {int(k): v for k, v in load_json(LABEL_MAP_PATH).items()}
label_to_text = load_json(LABEL_TO_TEXT_PATH)
num_classes = len(idx_to_label)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_LOAD_ERROR: Optional[str] = None
model: Optional[TCNClassifier] = None

try:
    if num_classes <= 0:
        raise RuntimeError(f"num_classes=0 (label_map.json 비었거나 없음) at {LABEL_MAP_PATH}")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL not found: {MODEL_PATH}")

    model = TCNClassifier(MODEL_FEAT_DIM, num_classes).to(device)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

except Exception as e:
    MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
    model = None

# ====== 요청/응답 ======
class HandsFrameRequest(BaseModel):
    features: List[float] = Field(default_factory=list)  # 126
    ts: Optional[int] = None
    framesReceived: Optional[int] = None
    mode: Optional[str] = None

class PredictResponse(BaseModel):
    label: str = ""
    text: str = ""
    confidence: float = 0.0
    frames_received: int = 0
    mode: str = "pending"   # buffering|pending|final|error
    streak: int = 0
    candidates: Optional[List[Tuple[str, float]]] = None

# ====== 손 프레임 버퍼(서버에서 30프레임 모음) ======
_buf: deque = deque(maxlen=T)
_last_label: Optional[str] = None
_streak: int = 0

router = APIRouter(prefix="/hands", tags=["hands"])

def pad_to_model_dim(vec126: np.ndarray) -> np.ndarray:
    """126 -> MODEL_FEAT_DIM(예: 336)로 패딩/절단 + 간단 스케일링"""
    v = vec126.astype(np.float32)

    # 픽셀 기반(0~640) 들어오면 대략 0~1대로 눌러주기
    if v.size:
        p95 = np.percentile(np.abs(v), 95)
        if p95 > 10.0:
            v = v / 1000.0

    if MODEL_FEAT_DIM == HAND_FEAT:
        return v

    if v.size < MODEL_FEAT_DIM:
        pad = np.zeros((MODEL_FEAT_DIM - v.size,), dtype=np.float32)
        return np.concatenate([v, pad], axis=0)

    return v[:MODEL_FEAT_DIM]

@router.get("/health")
def health():
    return {
        "ok": True,
        "device": str(device),
        "num_classes": num_classes,
        "T": T,
        "HAND_FEAT": HAND_FEAT,
        "MODEL_FEAT_DIM": MODEL_FEAT_DIM,
        "buf_len": len(_buf),
        "model_loaded": model is not None,
        "model_error": MODEL_LOAD_ERROR,
        "paths": {
            "LABEL_MAP_PATH": str(LABEL_MAP_PATH),
            "LABEL_TO_TEXT_PATH": str(LABEL_TO_TEXT_PATH),
            "MODEL_PATH": str(MODEL_PATH),
        }
    }

@router.post("/predict_frame", response_model=PredictResponse)
def predict_frame(req: HandsFrameRequest):
    global _last_label, _streak

    if model is None:
        return PredictResponse(
            mode="error",
            text=f"model not loaded: {MODEL_LOAD_ERROR}",
            frames_received=len(_buf),
            streak=_streak
        )

    raw126 = np.asarray(req.features or [], dtype=np.float32)
    if raw126.size != HAND_FEAT:
        return PredictResponse(
            mode="error",
            text=f"features length must be {HAND_FEAT} (got {raw126.size})",
            frames_received=len(_buf),
            streak=_streak
        )

    nonzero = int(np.count_nonzero(raw126))
    if nonzero < 10:
        _buf.clear()
        _last_label = None
        _streak = 0
        return PredictResponse(mode="buffering", frames_received=0, streak=0)

    v = pad_to_model_dim(raw126)
    _buf.append(v)

    if DEBUG and (len(_buf) % 10 == 0):
        print("[HANDS] buf_len", len(_buf), "nonzero", nonzero)

    if len(_buf) < T:
        return PredictResponse(mode="buffering", frames_received=len(_buf), streak=_streak)

    x = np.stack(list(_buf), axis=0).astype(np.float32)  # (T, F)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)     # (1,T,F)

    with torch.no_grad():
        logits = model(xt)[0]
        probs = torch.softmax(logits, dim=-1)

    top1_prob, top1_idx = torch.max(probs, dim=0)
    top1_prob = float(top1_prob.item())
    top1_idx = int(top1_idx.item())

    label = idx_to_label.get(top1_idx, str(top1_idx))
    text = label_to_text.get(label, label)

    k = min(5, probs.numel())
    top5_probs, top5_idxs = torch.topk(probs, k=k)
    candidates = [
        (idx_to_label.get(int(i), str(int(i))), float(p))
        for p, i in zip(top5_probs.tolist(), top5_idxs.tolist())
    ]

    if _last_label == label and top1_prob >= THRESHOLD:
        _streak += 1
    else:
        _last_label = label
        _streak = 1 if top1_prob >= THRESHOLD else 0

    mode = "final" if _streak >= STREAK_N else "pending"

    resp = PredictResponse(
        label=label,
        text=text,
        confidence=top1_prob,
        frames_received=len(_buf),
        mode=mode,
        streak=_streak,
        candidates=candidates,
    )

    if mode == "final":
        _buf.clear()
        _last_label = None
        _streak = 0

    return resp
