# “원본 JSON을 딥러닝이 먹기 쉬운 배열 파일로 바꾸는 코드”

import argparse
import json
from pathlib import Path
import numpy as np

T = 30
POINTS = 21 #손 랜드마크 점 
# 파일 일기
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def hand_keypoints_to_21x3(hand_kps):
    out = np.zeros((POINTS, 3), dtype=np.float32) # 0 0 0 빈 그릇 생성
    if not isinstance(hand_kps, list) or len(hand_kps) < POINTS * 3: # 길이가 63 보다 작으면 0 배열 반환
        return out
    arr = np.array(hand_kps[:POINTS * 3], dtype=np.float32).reshape(POINTS, 3)
    out[:, 0] = arr[:, 0]  # x
    out[:, 1] = arr[:, 1]  # y
    return out
    # 결과는 항상 21, 3 근데 왜 ?

def one_frame_json_to_tensor(j):
    frame = np.zeros((2, POINTS, 3), dtype=np.float32)

    if not isinstance(j, dict):
        return frame

    people = j.get("people", None) # 피플은 어디서 나오는건지? 

    # 1) people이 "사람 1명 dict"인 경우 (너 케이스)
    if isinstance(people, dict) and (
        "hand_left_keypoints_2d" in people or "hand_right_keypoints_2d" in people
    ):
        p0 = people

    # 2) people이 list인 경우: [person0, person1, ...]
    elif isinstance(people, list) and len(people) > 0 and isinstance(people[0], dict):
        p0 = people[0]

    # 3) people이 dict인데 {"0": person0, "1": person1} 같은 경우
    elif isinstance(people, dict) and len(people) > 0:
        # 값이 dict인 항목만 추려서 첫 번째 선택
        dict_items = [(k, v) for k, v in people.items() if isinstance(v, dict)]
        if not dict_items:
            return frame

        # 숫자키 우선 정렬 시도
        try:
            k0 = sorted([k for k, _ in dict_items], key=lambda x: int(str(x)))[0]
        except:
            k0 = dict_items[0][0]

        p0 = people[k0]

    else:
        return frame

    left = p0.get("hand_left_keypoints_2d", [])
    right = p0.get("hand_right_keypoints_2d", [])

    frame[0] = hand_keypoints_to_21x3(left)
    frame[1] = hand_keypoints_to_21x3(right)
    return frame

def folder_to_sample(folder: Path):
    json_files = sorted(folder.glob("*_keypoints.json"))
    if not json_files:
        return None, 0

    frames = []
    for fp in json_files:
        j = load_json(fp)
        ft = one_frame_json_to_tensor(j)  # ✅ 여기 이름 정확히!

        if ft.sum() == 0:
            continue

        frames.append(ft)

    received = len(frames)

    if received >= T:
        frames = frames[-T:]
    else:
        frames += [np.zeros((2, POINTS, 3), dtype=np.float32) for _ in range(T - received)]

    x = np.stack(frames, axis=0)
    return x, received


def extract_word_id(folder_name: str):
    for part in folder_name.split("_"):
        if part.startswith("WORD"):
            return part
    return "UNKNOWN"

def main(): # 이거 뭔지 물어봐야함 
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max", type=int, default=3)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in in_dir.iterdir() if p.is_dir()])

    done = 0
    for f in folders:
        x, received = folder_to_sample(f)
        if x is None:
            continue
                # dataset_raw 폴더는 어디서ㅏ 나는지 이건 내가 옵션 줄때 설정하는거 
        label = extract_word_id(f.name)
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        np.save(label_dir / f"{f.name}.npy", x)
        print(f"[OK] {f.name} shape={x.shape} framesReceived={received}")

        done += 1
        if done >= args.max:
            break

    print("DONE:", done)

if __name__ == "__main__":
    main()

#keypoints.json(여러 프레임) → (30,2,21,3)으로 정리 → 라벨 폴더(WORD1501) 아래 npy 저장 이 느낌으로 생각
# 즉 이 코드는 제이슨 모양 통일, 저장, 재료 만들기라고 생각하면 됌 