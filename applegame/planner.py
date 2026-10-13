"""게임 시작 때 보드를 한 번 읽고, 남는 시간 동안 머릿속에서 여러 판을 풀어 가장 좋은 수 순서를 고른다.

이 게임은 처음 보드가 정해지면 이후에 운이 없으므로(결정적), 미리 계산한 수 순서를 그대로 두면 그 점수가 나온다.

- rollout_beam: 매 수마다 (지금까지 점수 + 이 수 + 탐욕 롤아웃 점수)가 높은 상태 width개를 남겨 끝까지 진행.
  width=1이면 기존 롤아웃 탐색과 같다.
- noisy_playouts: 탐욕 규칙에 무작위성을 섞어 여러 판을 한꺼번에 두고 최고점을 고른다 (비교용).
- plan_game: 시간 예산 안에서 빔 폭을 늘려 가며 가장 좋은 수 순서를 반환.
"""
import time

import numpy as np
import torch

from .env import ROWS, COLS
from .policies import evaluate, greedy_min_policy


@torch.no_grad()
def rollout_beam(board, rects, width=4, deadline=None):
    """반환: (점수, [(r1, c1, r2, c2), ...]). deadline을 넘기면 지금까지 찾은 가장 좋은 완성 경로를 반환."""
    beam = [(board.clone(), 0, [])]
    best = (0, [])
    while beam:
        boards = torch.stack([b for b, _, _ in beam])
        mask, n = rects.legal(boards)
        children, parents, acts = [], [], []
        for i, (b, s, mv) in enumerate(beam):
            a = mask[i].nonzero().squeeze(1)
            if not len(a):
                if s > best[0]:
                    best = (s, mv)
                continue
            children.append(b[None].expand(len(a), -1, -1).masked_fill(rects.cover[a].view(-1, ROWS, COLS), 0))
            parents += [i] * len(a)
            acts.append(a)
        if not children:
            break
        child = torch.cat(children)
        acts = torch.cat(acts)
        parents = torch.tensor(parents)
        gain = n[parents, acts]
        acc = torch.tensor([beam[p][1] for p in parents.tolist()]) + gain
        # 같은 보드는 한 번만
        uniq, inv = torch.unique(child.flatten(1), dim=0, return_inverse=True)
        first = torch.full((len(uniq),), len(inv), dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(inv)), "amin")
        child, acts, parents, acc = child[first], acts[first], parents[first], acc[first]
        total = acc + evaluate(greedy_min_policy, child, rects)
        order = total.argsort(descending=True)[:width]
        beam = [(child[j], int(acc[j]), beam[int(parents[j])][2] + [tuple(rects.rects[acts[j]].tolist())])
                for j in order.tolist()]
        if deadline is not None and time.time() > deadline:
            # 시간 초과: 남은 빔 중 최선을 탐욕으로 마무리
            b, s, mv = beam[0]
            s2, mv2 = greedy_finish(b, rects)
            if s + s2 > best[0]:
                best = (s + s2, mv + mv2)
            break
    return best


@torch.no_grad()
def greedy_finish(board, rects):
    """탐욕 규칙으로 끝까지 두고 (점수, 수 목록) 반환."""
    b, score, moves = board.clone(), 0, []
    gen = torch.Generator().manual_seed(0)
    while True:
        mask, n = rects.legal(b[None])
        if not mask.any():
            return score, moves
        a = int(greedy_min_policy(b[None], mask, n, gen)[0])
        score += int(n[0, a])
        moves.append(tuple(rects.rects[a].tolist()))
        b = b.masked_fill(rects.cover[a].view(ROWS, COLS), 0)


@torch.no_grad()
def noisy_playouts(board, rects, num=512, noise=1.5, seed=0):
    """탐욕 + 무작위성으로 num판을 동시에 두고 (최고 점수, 수 목록) 반환."""
    gen = torch.Generator().manual_seed(seed)
    b = board[None].repeat(num, 1, 1)
    score = torch.zeros(num, dtype=torch.long)
    hist = [[] for _ in range(num)]
    alive = torch.arange(num)
    while len(alive):
        mask, n = rects.legal(b[alive])
        has = mask.any(1)
        alive, mask, n = alive[has], mask[has], n[has]
        if not len(alive):
            break
        pri = (-n.float() + torch.rand(mask.shape, generator=gen) * noise).masked_fill(~mask, -1e9)
        a = pri.argmax(1)
        score[alive] += n.gather(1, a[:, None]).squeeze(1)
        b[alive] = b[alive].masked_fill(rects.cover[a].view(-1, ROWS, COLS), 0)
        for g, act in zip(alive.tolist(), a.tolist()):
            hist[g].append(tuple(rects.rects[act].tolist()))
    k = int(score.argmax())
    return int(score[k]), hist[k]


def plan_game(board, rects, budget=50.0, log=print):
    """시간 예산(초) 안에서 가장 좋은 수 순서를 찾는다.
    1) 무작위 탐욕 512판으로 빠르게 안전한 답 확보
    2) 빔 폭 2, 4, 8...로 늘려 가되, 직전 실행 시간으로 예상해 예산 안에 끝날 때만 시작
       (중간에 끊긴 빔은 점수가 오히려 낮아서)
    3) 남는 시간은 무작위 판을 더 돌린다."""
    t0 = time.time()
    deadline = t0 + budget
    best = noisy_playouts(board, rects, 512, seed=0)
    log(f"  [계획] 무작위 512판 최고: {best[0]}점 ({time.time() - t0:.1f}초)")
    full = int((board > 0).sum())
    width, est = 2, None
    while best[0] < full:
        left = deadline - time.time()
        if est is not None and est > left:
            break
        t1 = time.time()
        s, mv = rollout_beam(board, rects, width, deadline)
        took = time.time() - t1
        log(f"  [계획] 빔 폭 {width}: {s}점 ({time.time() - t0:.1f}초)")
        if s > best[0]:
            best = (s, mv)
        est = took * 2.2      # 빔 폭을 2배로 하면 시간도 약 2배
        width *= 2
    seed = 1
    while best[0] < full and deadline - time.time() > 6:
        s, mv = noisy_playouts(board, rects, 512, seed=seed)
        seed += 1
        if s > best[0]:
            best = (s, mv)
            log(f"  [계획] 무작위 추가 512판: {s}점 ({time.time() - t0:.1f}초)")
    return best
