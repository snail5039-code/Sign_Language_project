<<<<<<< Updated upstream
from fastapi import FastAPI                 # FastAPI 서버 기능
from pydantic import BaseModel              # 요청 JSON을 DTO처럼 받게 해줌
from typing import Any, List                # 타입 표시용 (frames는 리스트)
from predict import SimpleClassifier, load_label_map, preprocess_like_train
from pathlib import Path
import numpy as np                          # 숫자 배열(모델 입력)을 만들 때 쓰는 라이브러리
import torch
import json


app = FastAPI()                             # FastAPI 서버 생성

# cpu로 모델 돌리고 모델 구조 만들고 학습시켜놨던 모델 가져오는 거임
device = "cpu"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "current"   # ✅ current 폴더에서만 모델 읽기
=======
# main.py
# ============================================================
# ✅ FastAPI 메인 서버 (너 프로젝트 구조 기준: main.py가 엔트리)
#
# 이 서버가 하는 일:
# 1) 프론트/스프링에서 (T=30) 프레임의 손+얼굴 좌표를 받는다.
# 2) 손(126) + 얼굴(1434) = 1560 차원으로 합쳐서 (30,1560) 입력을 만든다.
# 3) 학습된 TCN 모델(best_model_face.pth)을 통해 top1 라벨 + 확률(confidence) 계산
# 4) "후보 보여주기"는 안 하고,
#    - 확률이 높고(FINAL_TH 이상)
#    - 같은 라벨이 2번 연속 나오면(STREAK_N)
#    => mode="final"로 확정해서 반환
#    아니면 mode="pending"으로 아직 확정 못했다고 반환
#
# ✅ 이 방식이 '1초 더 보고 자동 확정' 전략임.
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json

import numpy as np
import torch

from train import TCNClassifier  # ✅ 너가 학습에 쓴 모델 클래스

# ------------------------------------------------------------
# 0) FastAPI 앱 객체
# ------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------
# 1) 경로/상수 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "current"   # ✅ 너 스샷에서 current 폴더가 모델 저장소

T = 30
HAND_POINTS = 21
FACE_POINTS = 478

# 손(2*21*3=126) + 얼굴(478*3=1434) = 1560
FEAT_DIM = 1560
>>>>>>> Stashed changes

# ------------------------------------------------------------
# 2) 자동 확정 규칙(서비스 전략)
# ------------------------------------------------------------
# base_th: 이 값 미만이면 "확신 낮음"으로 보고 streak 카운트 끊어버림
BASE_TH = 0.50

# final_th: 이 값 이상 + streak 조건 만족하면 최종 확정(final)
FINAL_TH = 0.65

# streak_n: 같은 라벨이 몇 번 연속 나오면 확정할지
STREAK_N = 2

# session 메모리 오래된 거 초기화
SESSION_TTL_SEC = 10

# ------------------------------------------------------------
# 3) label_to_text 로드 (라벨 -> 한국어 텍스트)
# ------------------------------------------------------------
def load_label_to_text() -> Dict[str, str]:
    path = MODEL_DIR / "label_to_text.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))    
    except Exception:
        return {}

label_to_text = load_label_to_text()

# ------------------------------------------------------------
# 4) label_map 로드 (정수 인덱스 <-> 라벨 문자열)
# ------------------------------------------------------------
def load_label_map() -> Dict[int, str]:
    """
    label_map.json은 train.py에서 저장한 idx_to_label 형태:
      {"0":"WORD0001","1":"WORD0002",...}
    json은 key가 문자열이라 int로 바꿔줌
    """
    p = MODEL_DIR / "label_map.json"
    if not p.exists():
        raise FileNotFoundError(f"label_map.json not found: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): v for k, v in d.items()}

idx_to_label = load_label_map()
label_to_idx = {v: k for k, v in idx_to_label.items()}

# ------------------------------------------------------------
# 5) 모델 로드 (TCNClassifier + best_model_face.pth)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

MODEL_PATH = MODEL_DIR / "best_model_face.pth"  # ✅ 너가 ren으로 저장한 손+얼굴 best
print("MODEL_PATH =", MODEL_PATH)
print("exists? =", MODEL_PATH.exists())
if not MODEL_PATH.exists():
    # 혹시 best_model_face.pth를 current 안에 안 넣었으면 여기서 터짐
    raise FileNotFoundError(f"model not found: {MODEL_PATH}")

model = TCNClassifier(feat_dim=FEAT_DIM, num_classes=len(label_to_idx)).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

<<<<<<< Updated upstream
class TranslateRequest(BaseModel):          # 요청(body) JSON 형식 정의 (요청 DTO)
    frames: List[Any]                       # frames는 배열(리스트). 안의 내용은 일단 Any(아무거나)

@app.post("/predict")                       # POST /predict 로 요청 오면 아래 함수 실행
def predict(req: TranslateRequest):         # req = 프론트가 보낸 요청(JSON)을 TranslateRequest로 만든 것
    frames = req.frames or []               # frames가 None일 수 있으니, 없으면 빈 리스트로 처리
    T = 30                                  # ✅ 우리가 고정할 프레임 길이(30프레임). (10fps면 약 3초)

    seq = []                                # 여기엔 "프레임 텐서"들을 차곡차곡 모을 거임

    for f in frames:                        # frames 안에 있는 프레임을 하나씩 꺼내서 f로 받음
        hands = f.get("hands") if isinstance(f, dict) else None  # f가 dict일 때만 hands 꺼냄
        if not hands:                       # hands가 없으면(손이 안 잡혔으면)
            continue                        # 이 프레임은 건너뜀

        frame_tensor = np.zeros((2, 21, 3), dtype=np.float32)    # ✅ 한 프레임을 (양손2, 점21, xyz3) 크기로 0으로 생성

        for hi in range(min(2, len(hands))): # hands가 1개면 0만, 2개면 0과 1까지 (최대 2손만 사용)
            hand = hands[hi]                # hi번째 손 데이터 꺼내기 (랜드마크 21개가 들어있는 리스트)
            if isinstance(hand, list) and len(hand) >= 21 and isinstance(hand[0], dict):  # 구조가 맞는지 체크
                pts = [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in hand[:21]]  # 21개 점을 [x,y,z]로 변환
                frame_tensor[hi] = np.array(pts, dtype=np.float32) # frame_tensor에 해당 손(0 or 1) 자리에 채워 넣기

        seq.append(frame_tensor)            # 완성된 프레임(2,21,3)을 seq에 추가

    framesReceived = len(seq)               # 손이 실제로 잡힌 프레임 개수(=seq에 들어간 개수)

    if framesReceived >= T:                 # 프레임이 30개 이상이면
        seq = seq[-T:]                      # 마지막 30개만 사용 (길이 맞추기)
    else:                                   # 프레임이 30개 미만이면
        pad_count = T - framesReceived      # 부족한 개수 계산
        pad_frames = [np.zeros((2, 21, 3), dtype=np.float32) for _ in range(pad_count)]  # 부족한 만큼 0프레임 만들기
        seq = seq + pad_frames              # 뒤에 0프레임을 붙여서 길이를 30으로 맞춤

    x = np.stack(seq, axis=0)               # ✅ 최종 입력 x 생성: (T,2,21,3) 즉 (30,2,21,3)
=======
# ------------------------------------------------------------
# 6) session별 상태 저장 (연속 확정용)
# ------------------------------------------------------------
# session_id -> {"last_label": str, "streak": int, "last_time": float}
SESSION_STATE: Dict[str, Dict[str, Any]] = {}

def cleanup_sessions():
    """오래된 session 상태 삭제(메모리 누수 방지)"""
    now = time.time()
    dead = [sid for sid, st in SESSION_STATE.items() if now - st["last_time"] > SESSION_TTL_SEC]
    for sid in dead:
        del SESSION_STATE[sid]
>>>>>>> Stashed changes

# ------------------------------------------------------------
# 7) 입력 (hand+face) -> (30,1560) 변환
# ------------------------------------------------------------
def make_tcn_input(seq_hand: List[np.ndarray], seq_face: List[np.ndarray]) -> np.ndarray:
    """
    seq_hand: 길이 T 리스트, 각 원소 shape=(2,21,3)
    seq_face: 길이 T 리스트, 각 원소 shape=(478,3)

<<<<<<< Updated upstream
=======
    반환: x shape=(T,1560)
    """
    # (T,2,21,3) -> (T,126)
    x_hand = np.stack(seq_hand, axis=0).astype(np.float32)
    hand_flat = x_hand.reshape(T, -1)  # (T,126)

    # (T,478,3) -> (T,1434)
    x_face = np.stack(seq_face, axis=0).astype(np.float32)
>>>>>>> Stashed changes

    # 얼굴은 카메라 이동에 민감할 수 있어서 상대좌표화(간단 버전)
    # 0번 포인트 기준으로 x,y를 빼준다
    anchor_xy = x_face[:, 0:1, :2]      # (T,1,2)
    x_face[:, :, :2] = x_face[:, :, :2] - anchor_xy

<<<<<<< Updated upstream
    print("x shape:", x.shape, "framesReceived:", framesReceived)  # 확인용(필요하면 켜기)
    print("hand0 sum:", x[:,0].sum(), "hand1 sum:", x[:,1].sum())
    # 조졌다 파이썬 개어렵다 ㅋㅋㅋㅋㅋㅋㅋㅋ 즉 ai가 먹기 좋게 모양을 일정하게 정리한거임!!
    
    # ===== 여기까지가 "AI 모델 들어가기 직전 단계" =====
    # 이제 모델이 생기면 model(x) 같은 걸 하면 됨

        # =========================================
    # 0) 손이 하나도 안 잡혔으면 모델 돌릴 필요 없음
    # =========================================
    if framesReceived == 0:
        return {
            "text": "번역 실패",
            "confidence": 0.0,
            "framesReceived": 0
        }

    # =========================================
    # 1) 학습 때랑 똑같이 전처리 적용 (중요!!)
    #    - 손목 기준 상대좌표
    #    - 스케일 나누기(정규화)
    #    - (predict.py에서 쓰던 함수 재사용)
    # =========================================
    x2 = preprocess_like_train(x)

    # =========================================
    # 2) numpy -> torch 텐서로 변환
    #    + unsqueeze(0)로 "배치 차원" 추가
    #    (30,2,21,3)  ->  (1,30,2,21,3)
    # =========================================
    xt = torch.from_numpy(x2).unsqueeze(0).to(device)

    # =========================================
    # 3) 예측은 학습이 아니라서 미분 계산 필요 없음
    #    no_grad() 쓰면 더 빠르고 메모리 덜 먹음
    # =========================================
    with torch.no_grad():
        # 모델에 넣어서 "점수(logits)" 얻기
        logits = model(xt)

        # 점수(logits)를 "확률(probs)"로 변환
        # softmax 하면 각 클래스 확률 합이 1이 됨
        # [0] 은 배치(1개) 중 첫 번째 데이터 꺼내는 것
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # =========================================
    # 4) 가장 확률이 큰 클래스 번호(idx) 찾기
    # =========================================
    pred_idx = int(probs.argmax())

    # =========================================
    # 5) idx -> 실제 라벨 문자열로 변환
    #    (idx_to_label은 label_map.json 기반)
    # =========================================
    label = idx_to_label[pred_idx]

    human_text = label_to_text.get(label, label) 
    # =========================================
    # 6) confidence = 그 라벨의 확률값(0~1)
    # =========================================
    confidence = float(probs[pred_idx])

    # =========================================
    # 7) 최종 응답 JSON 반환
    # =========================================
    return {
    "label": label,
    "text": human_text,
    "confidence": confidence,
    "framesReceived": framesReceived
=======
    face_flat = x_face.reshape(T, -1)   # (T,1434)

    # concat -> (T,1560)
    x = np.concatenate([hand_flat, face_flat], axis=1)

    # 학습 때 /1000 스케일 썼으니 동일하게 맞춤
    x = x / 1000.0
    return x

# ------------------------------------------------------------
# 8) FastAPI Request / Response 정의
# ------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    프론트/스프링에서 이 형태로 보내면 됨(예시):

    {
      "session_id": "room1_userA",
      "seq_hand": [ ... 30개 ... ],
      "seq_face": [ ... 30개 ... ]
>>>>>>> Stashed changes
    }

    seq_hand[t] = [[ [x,y,z]*21 ]*2 ]  형태(2손)
    seq_face[t] = [ [x,y,z]*478 ]
    """
    session_id: str
    seq_hand: List[Any]   # (30,2,21,3) 를 list로 받은 것
    seq_face: List[Any]   # (30,478,3)  를 list로 받은 것

<<<<<<< Updated upstream
    # 13) 최종 JSON 응답으로 반환 (스프링/프론트가 그대로 받음)
    # text = "안녕하세요!" if framesReceived >= 15 else "다시 해주세요"          # (임시) 프레임 수 기준 더미 로직
    # confidence = 0.77 if framesReceived >= 15 else 0.3                        # (임시) 신뢰도도 더미 값
=======
class PredictResponse(BaseModel):
    mode: str                 # "final" or "pending"
    label: Optional[str]      # 확정이면 label 채움, pending이면 None 가능
    text: Optional[str]       # label_to_text 있으면 한국어 매핑
    confidence: float         # top1 확률
    streak: int               # 현재 연속 카운트(디버그용)
>>>>>>> Stashed changes

    # return {"text": text, "confidence": confidence, "framesReceived": framesReceived}  # 최종 응답(JSON)

<<<<<<< Updated upstream
    class LabelMapItem(BaseModel):
        label: str
        text: str

    @app.post("/label-map")
    def upsert_label_map(item: LabelMapItem):
        global label_to_text
        label_to_text[item.label] = item.text

        path = MODEL_DIR / "label_to_text.json"
        path.write_text(json.dumps(label_to_text, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"ok": True, "count": len(label_to_text)}
=======
# ------------------------------------------------------------
# 9) 핵심: /predict
# ------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    print("=== /predict request ===")
    print(req)
    print("========================")
    """
    매 요청마다:
    1) (30,2,21,3) + (30,478,3) 받아서
    2) (30,1560)으로 만들고
    3) 모델 추론
    4) session 기준 연속 규칙으로 final/pending 결정
    """
    cleanup_sessions()

    sid = req.session_id

    # -------- 입력 길이 체크 --------
    if len(req.seq_hand) != T or len(req.seq_face) != T:
        return PredictResponse(
            mode="pending",
            label=None,
            text=None,
            confidence=0.0,
            streak=0,
        )

    # -------- numpy 변환 --------
    # JSON(list) -> np.ndarray
    seq_hand = [np.array(f, dtype=np.float32) for f in req.seq_hand]
    seq_face = [np.array(f, dtype=np.float32) for f in req.seq_face]

    # -------- (30,1560) 만들기 --------
    x = make_tcn_input(seq_hand, seq_face)                  # (30,1560)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)        # (1,30,1560)

    # -------- 모델 추론 --------
    with torch.no_grad():
        logits = model(xt)[0]                               # (C,)
        probs = torch.softmax(logits, dim=0)                # (C,)
        conf, pred_idx = torch.max(probs, dim=0)            # top1 확률 + 인덱스

    confidence = float(conf.item())
    pred_label = idx_to_label[int(pred_idx.item())]

    # --------------------------------------------------------
    # 연속 확정 로직
    # --------------------------------------------------------
    now = time.time()

    # session 상태 없으면 초기화
    st = SESSION_STATE.get(sid)
    if st is None:
        st = {"last_label": None, "streak": 0, "last_time": now}
        SESSION_STATE[sid] = st

    st["last_time"] = now

    # 1) confidence가 너무 낮으면: 확신 없음 -> streak 리셋
    if confidence < BASE_TH:
        st["last_label"] = None
        st["streak"] = 0
        return PredictResponse(
            mode="pending",
            label=None,
            text=None,
            confidence=confidence,
            streak=st["streak"],
        )

    # 2) confidence가 어느 정도 되면: streak 갱신
    if st["last_label"] == pred_label:
        st["streak"] += 1
    else:
        st["last_label"] = pred_label
        st["streak"] = 1

    # 3) 최종 확정 조건: streak>=2 AND confidence>=FINAL_TH
    if st["streak"] >= STREAK_N and confidence >= FINAL_TH:
        # 확정 후에는 streak를 초기화(다음 단어를 위해)
        st["last_label"] = None
        st["streak"] = 0

        text = label_to_text.get(pred_label)
        return PredictResponse(
            mode="final",
            label=pred_label,
            text=text,
            confidence=confidence,
            streak=STREAK_N,
        )

    # 아직 확정 못함
    return PredictResponse(
        mode="pending",
        label=None,
        text=None,
        confidence=confidence,
        streak=st["streak"],
    )
>>>>>>> Stashed changes
