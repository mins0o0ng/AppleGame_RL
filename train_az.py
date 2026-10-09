"""AlphaZero식 학습: MCTS 자기대국 -> (방문 분포, 남은 점수)로 지도학습 -> 반복.

예) python train_az.py --minutes 60
    python train_az.py --minutes 60 --channels 32 --blocks 3      # CPU
    python train_az.py --resume checkpoints_az/last.pt --minutes 60
"""
import argparse
import collections
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from applegame.env import ROWS, COLS, RectIndex
from applegame.mcts import VALUE_SCALE, BatchMCTS, mcts_play
from applegame.model import AppleNet
from applegame.policies import eval_boards, evaluate, model_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--iters", type=int, default=10**9)
    p.add_argument("--games", type=int, default=64, help="반복당 자기대국 판 수 (동시 진행)")
    p.add_argument("--sims", type=int, default=32, help="수당 MCTS 시뮬레이션 횟수")
    p.add_argument("--temp-moves", type=int, default=10, help="처음 몇 수는 방문 비율대로 샘플링 (탐험)")
    p.add_argument("--target", choices=["qmix", "visits"], default="qmix",
                   help="정책 정답: qmix = Gumbel MuZero식 Q 반영 개선 정책, visits = 방문 비율 (기존 AlphaZero)")
    p.add_argument("--c-scale", type=float, default=0.1, help="qmix에서 Q를 얼마나 강하게 반영할지")
    p.add_argument("--buffer", type=int, default=100_000, help="최근 국면 저장 개수")
    p.add_argument("--train-steps", type=int, default=200, help="반복당 학습 스텝")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--greedy-prior", type=float, default=20.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-mcts", type=int, default=32, help="MCTS 평가 판 수 (0이면 생략)")
    p.add_argument("--out", default="checkpoints_az")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def self_play(mcts, num_games, temp_moves, gen, target="qmix", c_scale=0.1):
    """num_games판을 동시에 끝까지 두고 (보드, 수 목록, 방문 분포, 남은 점수) 샘플을 반환."""
    boards = torch.randint(1, 10, (num_games, ROWS, COLS), generator=gen, dtype=torch.int8).numpy()
    roots = mcts.new_roots(boards)
    records = [[] for _ in roots]
    scores = np.zeros(num_games)
    moves = 0
    alive = [g for g in range(num_games) if not roots[g].terminal]
    while alive:
        visits = mcts.search([roots[g] for g in alive], add_noise=True)
        for g, n in zip(alive, visits):
            r = roots[g]
            visit_pi = n / n.sum()
            pi = r.improved_policy(c_scale) if target == "qmix" else visit_pi
            records[g].append((r.board, r.acts, pi.astype(np.float32), scores[g]))
            i = int(np.random.choice(len(n), p=visit_pi)) if moves < temp_moves else int(np.argmax(n))
            scores[g] += r.n_rm[i]
            roots[g] = mcts.advance(r, i)
        new = [roots[g] for g in alive if not roots[g].expanded]
        if new:
            mcts._evaluate(new)
        alive = [g for g in alive if not roots[g].terminal]
        moves += 1
    samples = [(b, a, pi, scores[g] - s) for g in range(num_games) for (b, a, pi, s) in records[g]]
    return samples, scores


def train_step(model, opt, rects, batch, dev):
    boards = torch.from_numpy(np.stack([s[0] for s in batch])).to(dev).view(-1, ROWS, COLS)
    target = torch.zeros(len(batch), rects.num_actions, device=dev)
    for j, (_, acts, pi, _) in enumerate(batch):
        target[j, torch.from_numpy(acts).to(dev)] = torch.from_numpy(pi).to(dev)
    z = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=dev) / VALUE_SCALE
    mask, n = rects.legal(boards)
    logits, value = model(boards, mask, n)
    pl = -(target * torch.log_softmax(logits, 1)).sum(1).mean()
    vl = F.mse_loss(value, z)
    opt.zero_grad()
    (pl + vl).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return pl.item(), vl.item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = args.device
    print(f"장치: {dev} | 모델 {args.channels}채널 x {args.blocks}블록 | 시뮬레이션 {args.sims} | 정답 {args.target}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    rects = RectIndex(device=dev)
    model = AppleNet(rects, args.channels, args.blocks, args.greedy_prior).to(dev)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    mcts = BatchMCTS(model, args.sims)
    buffer = collections.deque(maxlen=args.buffer)
    gen = torch.Generator().manual_seed(args.seed)
    val_boards = eval_boards(200, seed=999, device=dev)
    mcts_boards = eval_boards(max(args.eval_mcts, 1), seed=998)

    start, best = time.time(), -1.0
    for it in range(1, args.iters + 1):
        model.eval()
        t0 = time.time()
        samples, scores = self_play(mcts, args.games, args.temp_moves, gen, args.target, args.c_scale)
        buffer.extend(samples)
        t1 = time.time()

        model.train()
        pls, vls = [], []
        for _ in range(args.train_steps):
            pl, vl = train_step(model, opt, rects, random.sample(buffer, min(args.batch, len(buffer))), dev)
            pls.append(pl); vls.append(vl)
        elapsed = time.time() - start
        print(f"반복 {it:3d} | {elapsed/60:5.1f}분 | 자기대국 평균 {scores.mean():6.1f} ({t1-t0:4.0f}초) "
              f"| 정책손실 {np.mean(pls):.3f} | 가치손실 {np.mean(vls):.3f} | 버퍼 {len(buffer)}", flush=True)

        done = elapsed > args.minutes * 60
        if it % args.eval_every == 0 or done:
            model.eval()
            pol = evaluate(model_policy(model), val_boards, rects).float().mean().item()
            msg = f"  >> 검증: 정책 단독 {pol:.1f}"
            if args.eval_mcts:
                msg += f" | MCTS({args.sims}) {mcts_play(model, mcts_boards[:args.eval_mcts], args.sims).mean():.1f}"
            torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
            if pol > best:
                best = pol
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print(msg + f" | 정책 최고 {best:.1f}", flush=True)
        if done:
            break


if __name__ == "__main__":
    main()
