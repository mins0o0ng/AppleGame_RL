"""게임 스크린샷 -> 보드 인식 -> 롤아웃 탐색으로 둘 수 순서 출력.

예) python solve_image.py assets/board_1.png
    python solve_image.py my_screenshot.png --save-label   # 인식 결과를 .txt로 저장 (분포 수집용)

스크린샷은 사과가 모두 있는 시작 화면에서 게임판 영역 위주로 캡처하세요.
"""
import argparse
import os

import numpy as np
import torch

from applegame.env import RectIndex
from applegame.mcts import mcts_plan
from applegame.model import load_model
from applegame.policies import greedy_min_policy
from applegame.search import rollout_search
from applegame.vision import read_board


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--save-label", action="store_true")
    p.add_argument("--ckpt", default=None, help="학습된 모델 (있으면 MCTS, 없으면 롤아웃 탐색)")
    p.add_argument("--sims", type=int, default=64)
    args = p.parse_args()

    board, margin = read_board(args.image)
    print("인식한 보드:")
    for row in board:
        print(" ".join(str(int(v)) for v in row))
    weak = np.argwhere(margin < 1.0)
    if len(weak):
        print(f"!! 인식이 불확실한 칸 {len(weak)}개 (행, 열): {weak.tolist()}")
    print(f"합 {int(board.sum())} (10의 배수: {board.sum() % 10 == 0}) | "
          f"숫자별 개수 {np.bincount(board.ravel(), minlength=10)[1:].tolist()}")
    if args.save_label:
        out = os.path.splitext(args.image)[0] + ".txt"
        np.savetxt(out, board, fmt="%d")
        print("저장:", out)

    rects = RectIndex()
    if args.ckpt:
        model = load_model(args.ckpt, rects)
        score, moves = mcts_plan(model, torch.tensor(board), args.sims)
        name = f"MCTS({args.sims}회)"
    else:
        score, moves = rollout_search(torch.tensor(board), rects, greedy_min_policy, return_moves=True)
        name = "롤아웃 탐색"
    print(f"\n{name} 예상 점수: {score} / {int((board > 0).sum())}  ({len(moves)}수)")
    for i, (r1, c1, r2, c2) in enumerate(moves, 1):
        print(f"{i:3d}. ({r1},{c1}) -> ({r2},{c2})")


if __name__ == "__main__":
    main()
