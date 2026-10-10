"""배치형 사과게임 환경 (torch 텐서로 N개 보드를 동시에 진행).

- 행동: 10x17 보드 위의 모든 직사각형 (r1, c1, r2, c2), 총 8415개
- 유효 행동: 사과 합이 정확히 10이고, 네 변 모두에 사과가 하나 이상 걸친 '꽉 끼는' 직사각형
  (빈 칸만 더 포함한 사각형은 같은 수이므로 중복 제거)
- 보상: 지운 사과 개수 (= 실제 게임 점수)
- 종료: 더 이상 유효 행동이 없을 때
"""
import numpy as np
import torch

ROWS, COLS = 10, 17


def _build_actions(rows, cols):
    r1, c1, r2, c2 = [], [], [], []
    for a in range(rows):
        for b in range(cols):
            for c in range(a, rows):
                for d in range(b, cols):
                    r1.append(a); c1.append(b); r2.append(c); c2.append(d)
    return [torch.tensor(x) for x in (r1, c1, r2, c2)]


class RectIndex:
    """직사각형 인덱스 + 행렬곱 기반 사각형 합 계산 헬퍼.

    셀(170) x 행동(8415) 포함 행렬과의 행렬곱 한 번으로 모든 직사각형의 합을 구한다.
    (누적합 + gather 방식보다 CPU에서 약 15배 빠름)
    """

    def __init__(self, rows=ROWS, cols=COLS, device="cpu"):
        self.rows, self.cols, self.device = rows, cols, device
        r1, c1, r2, c2 = _build_actions(rows, cols)
        self.num_actions = len(r1)
        self.rects = torch.stack([r1, c1, r2, c2], 1)
        rr = torch.arange(rows).view(1, rows, 1)
        cc = torch.arange(cols).view(1, 1, cols)

        def cover_of(a, b, c, d):
            m = ((rr >= a.view(-1, 1, 1)) & (rr <= c.view(-1, 1, 1))
                 & (cc >= b.view(-1, 1, 1)) & (cc <= d.view(-1, 1, 1)))
            return m.view(len(a), -1)

        # 행동 -> 덮는 셀 마스크 (A, rows*cols)
        # (행렬은 모두 CPU에서 만든 뒤 마지막에 device로 옮긴다)
        cover = cover_of(r1, c1, r2, c2)
        self.cover = cover.to(device)
        self.cover_t = cover.float().T.contiguous().to(device)
        # 정책 헤드용: [내부 | 좌상 모서리 | 우하 모서리] 셀 -> 행동 행렬 (3*cells, A)
        cells = rows * cols
        tl = torch.zeros(cells, self.num_actions)
        br = torch.zeros(cells, self.num_actions)
        ar = torch.arange(self.num_actions)
        tl[r1 * cols + c1, ar] = 1
        br[r2 * cols + c2, ar] = 1
        self.head_t = torch.cat([cover.float().T, tl, br], 0).contiguous().to(device)
        # 윗변/아랫변/왼변/오른변 포함 행렬: '꽉 끼는' 사각형 판정용
        edges = [cover_of(r1, c1, r1, c2), cover_of(r2, c1, r2, c2),
                 cover_of(r1, c1, r2, c1), cover_of(r1, c2, r2, c2)]
        self.edges_t = [e.float().T.contiguous().to(device) for e in edges]

    def rect_sum(self, x):
        """x: (..., rows, cols) float -> (..., A) 각 직사각형 내부 합."""
        return x.flatten(-2) @ self.cover_t

    def legal(self, boards):
        """boards: (N, rows, cols) int -> (mask (N, A) bool, 지울 사과 수 (N, A) long)."""
        b = boards.flatten(1).float()
        s = b @ self.cover_t
        nz = (b > 0).float()
        legal = s == 10
        for e in self.edges_t:
            legal &= (nz @ e) > 0
        return legal, (nz @ self.cover_t).long()


def _composition(k, rng):
    """10을 1~9 사이 양의 정수 k개로 나누는 균등 무작위 조합."""
    while True:
        cuts = np.sort(rng.choice(np.arange(1, 10), k - 1, replace=False))
        parts = np.diff(np.concatenate([[0], cuts, [10]]))
        if (parts <= 9).all():
            return rng.permutation(parts)


def reverse_board(rng, max_h=3, max_w=4, sizes=(2, 3), size_p=(0.6, 0.4), frac=1.0):
    """실제 게임(기존 사과 디자인) 보드를 흉내 내는 역순 생성기.

    빈 칸만 있는 사각형(최대 max_h x max_w)을 고르고, 그 안 일부 칸(2~3개)에 합이 10인 숫자를 채우는 것을 반복한다.
    나중에 채운 묶음일수록 먼저 지우면 되므로, 이렇게 만든 묶음은 순서만 맞으면 모두 지울 수 있다.
    끝까지 못 채운 칸은 무작위로 채우되 전체 합이 10의 배수가 되게 한다.
    스크린샷 4장과 비교: 처음 가능 수 93 vs 88.5, 탐욕 159 vs 161.5, 작은 숫자 편중도 비슷함 (README 참고)."""
    b = np.zeros((ROWS, COLS), np.int64)
    fails = 0
    while fails < 500 and (b > 0).mean() < frac:
        h, w = rng.integers(1, max_h + 1), rng.integers(1, max_w + 1)
        k = rng.choice(sizes, p=size_p)
        if h * w < k:
            fails += 1
            continue
        r0, c0 = rng.integers(0, ROWS - h + 1), rng.integers(0, COLS - w + 1)
        region = b[r0:r0 + h, c0:c0 + w]
        if (region == 0).all():
            flat = region.ravel().copy()
            flat[rng.choice(h * w, k, replace=False)] = _composition(k, rng)
            b[r0:r0 + h, c0:c0 + w] = flat.reshape(h, w)
            fails = 0
        else:
            fails += 1
    left = b == 0
    while True:
        b[left] = rng.integers(1, 10, left.sum())
        if b.sum() % 10 == 0:
            return b


# 실제 게임 버전별 추정 생성 설정 (assets/board_*.png 분석, README 참고)
#   round: 둥근 사과 버전 - 보드 전체를 합 10 묶음으로 역순 생성 (작은 숫자 많음, 거의 다 지울 수 있음)
#   flat:  납작한 사과 버전 - 거의 균등 무작위 + 합 10의 배수, 묶음은 약 10%만
DISTS = {"round": dict(frac=1.0), "real": dict(frac=1.0), "flat": dict(frac=0.1)}


def sample_boards(k, gen=None, dist="uniform", device="cpu"):
    """보드 k개 생성.
    uniform:       1~9 균등 (기존)
    round (=real): 둥근 사과 버전 추정 생성 방식
    flat:          납작한 사과 버전 추정 생성 방식"""
    if dist == "uniform":
        return torch.randint(1, 10, (k, ROWS, COLS), generator=gen, device=device, dtype=torch.int8)
    if dist not in DISTS:
        raise ValueError(f"알 수 없는 분포: {dist}")
    gdev = gen.device if gen is not None else "cpu"
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gdev))
    rng = np.random.default_rng(seed)
    boards = np.stack([reverse_board(rng, **DISTS[dist]) for _ in range(k)])
    return torch.from_numpy(boards).to(torch.int8).to(device)


class BatchAppleEnv:
    def __init__(self, num_envs, device="cpu", seed=None, rects=None, dist="uniform"):
        self.n, self.device, self.dist = num_envs, device, dist
        self.rects = rects or RectIndex(device=device)
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.pool = None
        if dist != "uniform":
            # 생성이 느린 분포는 보드 묶음을 미리 만들어 두고 상하/좌우 뒤집기로 변형해 재사용
            self.pool = sample_boards(2048, self.gen, dist, device)
        self.boards = self._new_boards(num_envs)
        self.score = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._refresh()

    def _new_boards(self, k):
        if self.pool is None:
            return sample_boards(k, self.gen, self.dist, self.device)
        idx = torch.randint(0, len(self.pool), (k,), generator=self.gen, device=self.device)
        b = self.pool[idx]
        flip = torch.randint(0, 4, (k,), generator=self.gen, device=self.device)
        b = torch.where((flip % 2 == 1)[:, None, None], b.flip(1), b)
        return torch.where((flip >= 2)[:, None, None], b.flip(2), b)

    def _refresh(self):
        self.mask, self.n_removed = self.rects.legal(self.boards)

    def step(self, actions):
        """actions: (N,) long. 반환: reward (N,), done (N,), 종료된 판의 최종 점수 (N,, 미종료는 -1)."""
        reward = self.n_removed.gather(1, actions[:, None]).squeeze(1)
        cover = self.rects.cover[actions].view(-1, ROWS, COLS)
        self.boards = self.boards.masked_fill(cover, 0)
        self.score += reward
        self._refresh()
        done = ~self.mask.any(1)
        final = torch.where(done, self.score, torch.full_like(self.score, -1))
        if done.any():
            idx = done.nonzero().squeeze(1)
            self.boards[idx] = self._new_boards(len(idx))
            self.score[idx] = 0
            self.mask[idx], self.n_removed[idx] = self.rects.legal(self.boards[idx])
        return reward.float(), done, final


def encode(boards):
    """(N, rows, cols) int -> (N, 10, rows, cols) 원-핫 float (0=빈칸, 1~9)."""
    return torch.nn.functional.one_hot(boards.long(), 10).permute(0, 3, 1, 2).float()
