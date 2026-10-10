"""배치형 사과게임 환경 (torch 텐서로 N개 보드를 동시에 진행).

- 행동: 10x17 보드 위의 모든 직사각형 (r1, c1, r2, c2), 총 8415개
- 유효 행동: 사과 합이 정확히 10이고, 네 변 모두에 사과가 하나 이상 걸친 '꽉 끼는' 직사각형
  (빈 칸만 더 포함한 사각형은 같은 수이므로 중복 제거)
- 보상: 지운 사과 개수 (= 실제 게임 점수)
- 종료: 더 이상 유효 행동이 없을 때
"""
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


# 실제 게임 스크린샷(assets/sample_board.txt)에서 센 숫자 1~9 개수 (+1 평활화). 한 장 기준 추정치.
REAL_DIGIT_COUNTS = [24, 32, 20, 22, 11, 20, 19, 12, 10]


def sample_boards(k, gen=None, dist="uniform", device="cpu"):
    """보드 k개 생성.
    uniform: 1~9 균등 (기존)
    real:    실제 게임에서 추정한 숫자 비율 + 전체 합이 10의 배수인 보드만 채택"""
    if dist == "uniform":
        return torch.randint(1, 10, (k, ROWS, COLS), generator=gen, device=device, dtype=torch.int8)
    if dist != "real":
        raise ValueError(f"알 수 없는 분포: {dist}")
    w = torch.tensor(REAL_DIGIT_COUNTS, dtype=torch.float, device=device) + 1
    out = []
    need = k
    while need > 0:
        cand = torch.multinomial(w, (need * 12 + 16) * ROWS * COLS, replacement=True, generator=gen)
        cand = (cand + 1).view(-1, ROWS, COLS).to(torch.int8)
        ok = cand.long().sum((1, 2)) % 10 == 0
        out.append(cand[ok][:need])
        need -= len(out[-1])
    return torch.cat(out)


class BatchAppleEnv:
    def __init__(self, num_envs, device="cpu", seed=None, rects=None, dist="uniform"):
        self.n, self.device, self.dist = num_envs, device, dist
        self.rects = rects or RectIndex(device=device)
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.boards = self._new_boards(num_envs)
        self.score = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._refresh()

    def _new_boards(self, k):
        return sample_boards(k, self.gen, self.dist, self.device)

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
