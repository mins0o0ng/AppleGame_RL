"""고정된 평가 보드에서 기준선과 학습 모델을 비교.

예) python evaluate.py --ckpt checkpoints/best.pt --boards 1000
    python evaluate.py --ckpt checkpoints/best.pt --boards 100 --beam 16
    python evaluate.py --ckpt checkpoints_az/best.pt --boards 100 --mcts 64
"""
import argparse
import time

import torch

from applegame.env import RectIndex
from applegame.model import AppleNet
from applegame.policies import eval_boards, evaluate, greedy_min_policy, model_policy, random_policy
from applegame.mcts import mcts_play
from applegame.search import beam_search, rollout_search


def report(name, scores, sec):
    s = scores.float()
    print(f"{name:22s} 평균 {s.mean():6.1f} ± {s.std() / len(s) ** 0.5:.1f} | 최고 {s.max():4.0f} "
          f"| 120점 이상 {(s >= 120).float().mean() * 100:5.1f}% | 판당 {sec / len(s) * 1000:.0f}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--boards", type=int, default=1000)
    p.add_argument("--beam", type=int, default=0, help="빔 폭 (0이면 빔 서치 생략)")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--mcts", type=int, default=0, help="MCTS 시뮬레이션 횟수 (0이면 생략)")
    p.add_argument("--rollout", type=int, default=0, help="롤아웃 탐색으로 평가할 보드 수 (느림, 0이면 생략)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    print(f"장치: {args.device}", flush=True)

    rects = RectIndex(device=args.device)
    boards = eval_boards(args.boards, device=args.device)
    runs = [("무작위(유효 수)", random_policy), ("탐욕(최소 제거)", greedy_min_policy)]
    model = None
    if args.ckpt:
        model = AppleNet(rects, args.channels, args.blocks).to(args.device)
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
        model.eval()
        runs.append(("정책망 단독", model_policy(model)))
    for name, pol in runs:
        t = time.time()
        report(name, evaluate(pol, boards, rects), time.time() - t)
    if args.rollout:
        sub = boards[:args.rollout]
        t = time.time()
        report("탐욕 기준 비교(부분집합)", evaluate(greedy_min_policy, sub, rects), time.time() - t)
        t = time.time()
        scores = torch.tensor([rollout_search(b, rects, greedy_min_policy) for b in sub])
        report("롤아웃 탐색(탐욕)", scores, time.time() - t)
    if model is not None and args.mcts:
        t = time.time()
        report(f"MCTS({args.mcts}회)", mcts_play(model, boards, args.mcts), time.time() - t)
    if model is not None and args.beam:
        t = time.time()
        scores = torch.tensor([beam_search(model, b, args.beam, args.topk) for b in boards])
        report(f"가치망+빔서치(W={args.beam})", scores, time.time() - t)


if __name__ == "__main__":
    main()
