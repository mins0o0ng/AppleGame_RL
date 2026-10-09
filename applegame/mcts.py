"""AlphaZero식 MCTS (1인용, 결정적 게임).

- 정책망 출력 = 각 수의 사전확률 P, 가치망 출력 x10 = 이 국면에서 앞으로 지울 사과 수 예측
- Q(s, a) = 지운 사과 수 + 자식 국면의 가치, PUCT로 선택
- 점수 단위가 판마다 달라서 Q는 노드별 최솟값/최댓값으로 0~1 정규화 (MuZero 방식)
- 여러 판을 동시에 탐색하고, 잎 노드 평가는 한 번의 배치 추론으로 처리
"""
import math

import numpy as np
import torch

from .env import ROWS, COLS

VALUE_SCALE = 10.0


class Node:
    __slots__ = ("board", "acts", "prior", "prior0", "n_rm", "N", "W", "children", "value", "expanded", "terminal")

    def __init__(self, board):
        self.board = board          # (rows*cols,) int8 numpy
        self.expanded = False
        self.terminal = False
        self.children = {}
        self.value = 0.0

    def expand(self, acts, prior, n_rm, value):
        self.acts, self.prior, self.prior0, self.n_rm = acts, prior, prior, n_rm
        self.N = np.zeros(len(acts), dtype=np.float32)
        self.W = np.zeros(len(acts), dtype=np.float32)
        self.value = value
        self.expanded = True
        self.terminal = len(acts) == 0

    def q_normalized(self):
        """방문한 수는 평균 Q, 아직 안 간 수는 이 국면의 가치 추정치로 채운 뒤 0~1 정규화."""
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), self.value)
        lo, hi = min(q.min(), self.value), max(q.max(), self.value)
        return (q - lo) / (hi - lo) if hi > lo else np.zeros_like(q)

    def select(self, c_puct):
        u = c_puct * self.prior * math.sqrt(self.N.sum() + 1) / (1 + self.N)
        return int(np.argmax(self.q_normalized() + u))

    def improved_policy(self, c_scale, c_visit=50.0):
        """Gumbel MuZero식 개선 정책: softmax(log P + (c_visit + max N) * c_scale * Q정규화).
        방문 비율보다 Q 정보를 직접 반영하므로 시뮬레이션이 적어도 학습 신호가 선명하다."""
        logits = np.log(self.prior0 + 1e-12) + (c_visit + self.N.max()) * c_scale * self.q_normalized()
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()


class BatchMCTS:
    def __init__(self, model, sims=32, c_puct=1.25, dirichlet=0.3, noise_frac=0.25):
        self.model = model
        self.rects = model.rects
        self.dev = model.rects.device
        self.cover = model.rects.cover.cpu().numpy()
        self.sims, self.c_puct = sims, c_puct
        self.dirichlet, self.noise_frac = dirichlet, noise_frac

    @torch.no_grad()
    def _evaluate(self, nodes):
        """잎 노드들을 한 번에 평가하고 펼친다."""
        boards = torch.from_numpy(np.stack([n.board for n in nodes])).to(self.dev).view(-1, ROWS, COLS)
        mask, n = self.rects.legal(boards)
        logits, value = self.model(boards, mask, n)
        logits, value = logits.cpu(), value.cpu().numpy() * VALUE_SCALE
        mask, n = mask.cpu(), n.cpu()
        for i, node in enumerate(nodes):
            acts = mask[i].nonzero().squeeze(1)
            if len(acts):
                prior = torch.softmax(logits[i, acts], 0).numpy()
                node.expand(acts.numpy(), prior, n[i, acts].numpy().astype(np.float32), float(value[i]))
            else:
                node.expand(acts.numpy(), np.zeros(0, np.float32), np.zeros(0, np.float32), 0.0)

    def _child(self, node, i):
        child = node.children.get(i)
        if child is None:
            b = node.board.copy()
            b[self.cover[node.acts[i]]] = 0
            child = node.children[i] = Node(b)
        return child

    def new_roots(self, boards):
        roots = [Node(b.reshape(-1).copy()) for b in boards]
        self._evaluate(roots)
        return roots

    def search(self, roots, add_noise=False):
        """각 루트에서 sims번 시뮬레이션. 반환: 루트별 방문 횟수 (len(acts),)."""
        if add_noise:
            for r in roots:
                if not r.terminal:
                    eta = np.random.dirichlet([self.dirichlet] * len(r.acts)).astype(np.float32)
                    r.prior = (1 - self.noise_frac) * r.prior + self.noise_frac * eta
        for _ in range(self.sims):
            paths, leaves = [], []
            for r in roots:
                if r.terminal:
                    continue
                node, path = r, []
                while node.expanded and not node.terminal:
                    i = node.select(self.c_puct)
                    path.append((node, i))
                    node = self._child(node, i)
                paths.append(path)
                leaves.append(node)
            if not leaves:
                break
            todo = [n for n in leaves if not n.expanded]
            if todo:
                self._evaluate(list({id(n): n for n in todo}.values()))
            for path, leaf in zip(paths, leaves):
                g = 0.0 if leaf.terminal else leaf.value
                for node, i in reversed(path):
                    g += node.n_rm[i]
                    node.N[i] += 1
                    node.W[i] += g
        return [r.N.copy() if not r.terminal else None for r in roots]

    def advance(self, root, i):
        """수 i를 두고 해당 자식을 새 루트로 (서브트리 재사용)."""
        return self._child(root, i)


@torch.no_grad()
def mcts_play(model, boards, sims=32):
    """평가용: 노이즈 없이 방문 수 최대인 수를 두며 끝까지. 반환: 판별 점수 (N,)."""
    mcts = BatchMCTS(model, sims)
    roots = mcts.new_roots(boards.cpu().numpy())
    scores = np.zeros(len(roots))
    alive = list(range(len(roots)))
    while alive:
        mcts.search([roots[g] for g in alive])
        nxt = []
        for g in alive:
            r = roots[g]
            if r.terminal:
                continue
            i = int(np.argmax(r.N))
            scores[g] += r.n_rm[i]
            roots[g] = mcts.advance(r, i)
            nxt.append(g)
        new = [roots[g] for g in nxt if not roots[g].expanded]
        if new:
            mcts._evaluate(new)
        alive = [g for g in nxt if not roots[g].terminal]
    return torch.tensor(scores)
