"""게임 시작 때 보드를 한 번 읽고, 남는 시간 동안 머릿속에서 여러 판을 풀어 가장 좋은 수 순서를 고른다.

이 게임은 처음 보드가 정해지면 이후에 운이 없으므로(결정적), 미리 계산한 수 순서를 그대로 두면 그 점수가 나온다.

- rollout_beam: 매 수마다 (지금까지 점수 + 이 수 + 탐욕 롤아웃 점수)가 높은 상태 width개를 남겨 끝까지 진행.
  width=1이면 기존 롤아웃 탐색과 같다.
- noisy_playouts: 탐욕 규칙에 무작위성을 섞어 여러 판을 한꺼번에 두고 최고점을 고른다 (비교용).
- plan_game: 시간 예산 안에서 빔 폭을 늘려 가며 가장 좋은 수 순서를 반환.
- mobility_beam: 롤아웃 없이 (점수 + lam x 남은 유효 수)로 넓은 빔. 실험용 (README "다른 롤아웃 규칙" 참고).

탐욕 대신 다른 규칙도 시험했다 (실제 보드 5장, 롤아웃 빔 폭 4 기준 평균):
  탐욕 137.6 / 개선 규칙(겹침 감점 + 큰 숫자 가점) 138.0 / 선택지 빔 132.8.
  보드마다 이기는 방법은 달랐지만, 120초 예산 안에서는 여러 방법을 섞은 조합(136.8)이
  탐욕 빔 폭 4를 시드만 바꿔 반복하는 쪽(138.2)보다 낮아서 기본값은 후자로 둔다.
"""
import time

import numpy as np
import torch

from .env import ROWS, COLS
from .policies import evaluate, greedy_min_policy


@torch.no_grad()
def rollout_beam(board, rects, width=4, deadline=None, seed=0, rollout=None):
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
        total = acc + evaluate(rollout or greedy_min_policy, child, rects, seed=seed)
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


def plan_game(board, rects, budget=50.0, log=print, init=None):
    """시간 예산(초) 안에서 가장 좋은 수 순서를 찾는다.
    1) 무작위 탐욕 512판으로 빠르게 안전한 답 확보 (init로 이미 구한 답을 넘길 수 있음)
    2) 빔 폭 2 -> 빔 폭 4를 시드(롤아웃 동점 처리 순서)를 바꿔 가며 반복.
       실험상 빔을 8로 넓히는 것보다 폭 4를 여러 시드로 돌리는 쪽이 시간 대비 효과가 좋았다.
       직전 실행 시간으로 예상해 예산 안에 끝날 때만 시작한다 (중간에 끊긴 빔은 점수가 낮음).
    3) 남는 시간은 무작위 판을 더 돌린다."""
    t0 = time.time()
    deadline = t0 + budget
    best = init or noisy_playouts(board, rects, 512, seed=0)
    log(f"  [계획] 예산 {budget:.0f}초 | 시작 답: {best[0]}점")
    full = int((board > 0).sum())
    schedule = [(2, 0)] + [(4, sd) for sd in range(100)]
    est = {}
    for width, seed in schedule:
        if best[0] >= full:
            break
        left = deadline - time.time()
        guess = est.get(width, est.get(width // 2, 0) * 2.2 if est else None)
        if guess is not None and guess > left:
            break
        t1 = time.time()
        s, mv = rollout_beam(board, rects, width, deadline, seed=seed)
        est[width] = time.time() - t1
        mark = ""
        if s > best[0]:
            best, mark = (s, mv), " <- 최고"
        log(f"  [계획] 빔 폭 {width} 시드 {seed}: {s}점{mark} ({time.time() - t0:.0f}초)")
    seed = 1
    while best[0] < full and deadline - time.time() > 6:
        s, mv = noisy_playouts(board, rects, 512, seed=seed)
        seed += 1
        if s > best[0]:
            best = (s, mv)
            log(f"  [계획] 무작위 추가 512판: {s}점 <- 최고 ({time.time() - t0:.0f}초)")
    return best


@torch.no_grad()
def mobility_beam(board, rects, width=200, lam=1.0, deadline=None):
    """롤아웃 없이 (지금까지 점수 + lam x 둔 뒤 남는 유효 수 개수)로 상위 width개 상태를 남기며 끝까지 진행.
    '선택지를 많이 남기는 수'가 탐욕 규칙보다 강했던 실험 결과를 넓은 빔으로 활용한다.
    반환: (점수, [(r1, c1, r2, c2), ...])"""
    beam_b = board[None].clone()
    beam_s = torch.zeros(1, dtype=torch.long)
    beam_m = [[]]
    best = (0, [])
    mask, n = rects.legal(beam_b)
    while len(beam_b):
        has = mask.any(1)
        for i in (~has).nonzero().squeeze(1).tolist():
            if int(beam_s[i]) > best[0]:
                best = (int(beam_s[i]), beam_m[i])
        if not has.any():
            break
        idx = mask.nonzero()                              # (K, 2): (부모, 행동)
        par, act = idx[:, 0], idx[:, 1]
        child = beam_b[par].masked_fill(rects.cover[act].view(-1, ROWS, COLS), 0)
        acc = beam_s[par] + n[par, act]
        uniq, inv = torch.unique(child.flatten(1), dim=0, return_inverse=True)
        first = torch.full((len(uniq),), len(inv), dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(inv)), "amin")
        child, par, act, acc = child[first], par[first], act[first], acc[first]
        cm, cn = rects.legal(child)
        val = acc.float() + lam * cm.sum(1).float()
        order = val.argsort(descending=True)[:width]
        beam_m = [beam_m[int(par[j])] + [tuple(rects.rects[int(act[j])].tolist())] for j in order.tolist()]
        beam_b, beam_s, mask, n = child[order], acc[order], cm[order], cn[order]
        if deadline is not None and time.time() > deadline:
            for i in range(len(beam_b)):   # 시간 초과: 남은 빔을 탐욕으로 마무리
                s2, mv2 = greedy_finish(beam_b[i], rects)
                if int(beam_s[i]) + s2 > best[0]:
                    best = (int(beam_s[i]) + s2, beam_m[i] + mv2)
            break
    return best


def plan_game_fast(board, rects=None, budget=50.0, log=print):
    """numba 버전 플래너 (기존 torch 버전보다 약 60배 빠름).
    1) 롤아웃 빔 폭 16으로 첫 답 (1~2초)  2) 남은 시간 동안 폭 32를 시드를 바꿔 반복.
    실제 보드 5장, 판당 약 37초: 폭 32 x 시드 12개 최고 평균 142.8 (폭 64 x 6개 142.2, 폭 512 1회 141.6)."""
    from . import fast
    b = np.asarray(board, np.int8).reshape(-1)
    full = int((b > 0).sum())
    t0 = time.time()
    deadline = t0 + budget
    best = fast.beam_search(b, 16, 0, deadline=deadline, seed=0)
    log(f"  [계획] 예산 {budget:.0f}초 | 빔 16: {best[0]}점 ({time.time() - t0:.1f}초)")
    seed, took = 1, None
    while best[0] < full:
        left = deadline - time.time()
        if took is not None and took * 1.2 > left:
            break
        t1 = time.time()
        s, mv = fast.beam_search(b, 32, 0, deadline=deadline, seed=seed)
        took = time.time() - t1
        if s > best[0]:
            best = (s, mv)
            log(f"  [계획] 빔 32 시드 {seed}: {s}점 <- 최고 ({time.time() - t0:.0f}초)")
        seed += 1
    log(f"  [계획] 빔 32 {seed - 1}회 실행")
    return best


def warmup():
    """numba 첫 컴파일(수십 초 걸릴 수 있음)을 게임 시작 전에 끝낸다. numba가 없으면 False."""
    try:
        from . import fast
    except ImportError:
        return False
    b = np.random.default_rng(0).integers(1, 10, 170).astype(np.int8)
    fast.beam_search(b, 2, 0)
    fast.eval_children(np.stack([b, b]), 0, 1, 0.01)
    return True
