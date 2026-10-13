"""numba로 빠르게 만든 핵심 연산 (단일 보드 기준, CPU 멀티코어).

torch 버전(env.RectIndex.legal)은 매번 사각형 8415개를 모두 계산한다.
여기서는 왼쪽 위 칸을 고정하고 사각형을 키워 가다가 합이 10을 넘으면 바로 멈춘다 (숫자가 모두 양수라 합은 줄지 않음).
실제로 확인하는 사각형이 크게 줄어든다.

보드는 길이 170(10x17)의 int8 배열. 수는 (r1, c1, r2, c2, 지울 사과 수) 5개 정수.
"""
import numpy as np
from numba import njit, prange

R, C = 10, 17
MAXM = 1024   # 한 보드의 유효 수 최대 개수 (실제로는 보통 200 이하)


@njit(cache=True)
def _prefix(b):
    s = np.zeros((R + 1, C + 1), np.int32)
    n = np.zeros((R + 1, C + 1), np.int32)
    for r in range(R):
        for c in range(C):
            v = b[r * C + c]
            s[r + 1, c + 1] = s[r, c + 1] + s[r + 1, c] - s[r, c] + v
            n[r + 1, c + 1] = n[r, c + 1] + n[r + 1, c] - n[r, c] + (1 if v > 0 else 0)
    return s, n


@njit(cache=True, inline="always")
def _rect(p, r1, c1, r2, c2):
    return p[r2 + 1, c2 + 1] - p[r1, c2 + 1] - p[r2 + 1, c1] + p[r1, c1]


@njit(cache=True)
def legal_moves(b, out):
    """b의 유효 수(합 10, 네 변에 사과가 걸친 꽉 끼는 사각형)를 out(MAXM, 5)에 채우고 개수를 반환."""
    s, n = _prefix(b)
    m = 0
    for r1 in range(R):
        for c1 in range(C):
            for r2 in range(r1, R):
                if _rect(s, r1, c1, r2, c1) > 10:
                    break                       # 첫 열부터 10 초과면 더 아래로 늘려도 초과
                for c2 in range(c1, C):
                    t = _rect(s, r1, c1, r2, c2)
                    if t > 10:
                        break
                    if t == 10:
                        if (_rect(n, r1, c1, r1, c2) > 0 and _rect(n, r2, c1, r2, c2) > 0
                                and _rect(n, r1, c1, r2, c1) > 0 and _rect(n, r1, c2, r2, c2) > 0):
                            out[m, 0] = r1
                            out[m, 1] = c1
                            out[m, 2] = r2
                            out[m, 3] = c2
                            out[m, 4] = _rect(n, r1, c1, r2, c2)
                            m += 1
    return m


@njit(cache=True, inline="always")
def apply_move(b, mv):
    for r in range(mv[0], mv[2] + 1):
        for c in range(mv[1], mv[3] + 1):
            b[r * C + c] = 0


@njit(cache=True)
def greedy_rollout(b0, seed):
    """탐욕(사과를 가장 적게 지우는 수, 동점은 무작위)으로 끝까지 두고 점수 반환. b0는 바꾸지 않음."""
    np.random.seed(seed)
    b = b0.copy()
    buf = np.empty((MAXM, 5), np.int32)
    score = 0
    while True:
        m = legal_moves(b, buf)
        if m == 0:
            return score
        best, bestv = -1, 1e9
        for i in range(m):
            v = buf[i, 4] + np.random.random() * 0.1
            if v < bestv:
                bestv, best = v, i
        score += buf[best, 4]
        apply_move(b, buf[best])


@njit(cache=True)
def mobility_rollout(b0, seed, w_n):
    """'둔 뒤 남는 유효 수가 많은 수' 우선 (동점이면 적게 지우는 수)으로 끝까지 두고 점수 반환.
    torch에서는 20배 느려 못 쓰던 규칙. 우선순위 = 남는 유효 수 - w_n x 지울 사과 수."""
    np.random.seed(seed)
    b = b0.copy()
    buf = np.empty((MAXM, 5), np.int32)
    buf2 = np.empty((MAXM, 5), np.int32)
    score = 0
    while True:
        m = legal_moves(b, buf)
        if m == 0:
            return score
        best, bestv = -1, -1e9
        for i in range(m):
            c = b.copy()
            apply_move(c, buf[i])
            v = legal_moves(c, buf2) - w_n * buf[i, 4] + np.random.random() * 1e-3
            if v > bestv:
                bestv, best = v, i
        score += buf[best, 4]
        apply_move(b, buf[best])


@njit(parallel=True, cache=True)
def eval_children(children, seed, policy, w_n):
    """여러 보드(K, 170)를 각각 롤아웃해 점수 배열 반환. 코어마다 나눠 병렬 실행.
    policy 0 = 탐욕, 1 = 선택지 많이 남기기."""
    k = children.shape[0]
    out = np.empty(k, np.int32)
    for i in prange(k):
        if policy == 0:
            out[i] = greedy_rollout(children[i], seed + i)
        else:
            out[i] = mobility_rollout(children[i], seed + i, w_n)
    return out


def moves_of(board):
    """numpy 보드(10x17 또는 170) -> 유효 수 배열 (m, 5)."""
    b = np.ascontiguousarray(np.asarray(board, np.int8).reshape(-1))
    buf = np.empty((MAXM, 5), np.int32)
    m = legal_moves(b, buf)
    return buf[:m].copy()


def beam_search(board, width=16, policy=0, w_n=0.01, deadline=None, seed=0):
    """롤아웃 빔 서치 (numba 버전). 매 수마다 (지금까지 점수 + 이 수 + 롤아웃 점수)가 높은 상태 width개를 남긴다.
    policy 0 = 탐욕 롤아웃, 1 = 선택지 많이 남기기 롤아웃.
    반환: (점수, [(r1, c1, r2, c2), ...])"""
    import time
    b0 = np.ascontiguousarray(np.asarray(board, np.int8).reshape(-1))
    beam = [(b0, 0, [])]
    best = (0, [])
    buf = np.empty((MAXM, 5), np.int32)
    step = 0
    while beam:
        kids, meta, seen = [], [], set()
        for bi, (b, s, mv) in enumerate(beam):
            m = legal_moves(b, buf)
            if m == 0:
                if s > best[0]:
                    best = (s, mv)
                continue
            for i in range(m):
                c = b.copy()
                apply_move(c, buf[i])
                key = c.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                kids.append(c)
                meta.append((bi, tuple(int(x) for x in buf[i, :4]), int(buf[i, 4])))
        if not kids:
            break
        roll = eval_children(np.stack(kids), seed * 1000003 + step * 7919, policy, w_n)
        acc = np.array([beam[bi][1] + g for bi, _, g in meta])
        total = acc + roll
        order = np.argsort(-total, kind="stable")[:width]
        if total[order[0]] > best[0]:
            # 이 자식 + 롤아웃으로 실제 달성 가능한 점수 (끝까지 가면 확정)
            pass
        beam = [(kids[j], int(acc[j]), beam[meta[j][0]][2] + [meta[j][1]]) for j in order]
        step += 1
        if deadline is not None and time.time() > deadline:
            for b, s, mv in beam:            # 시간 초과: 남은 빔을 탐욕으로 마무리
                s2, mv2 = greedy_finish(b)
                if s + s2 > best[0]:
                    best = (s + s2, mv + mv2)
            break
    return best


def greedy_finish(board):
    """탐욕으로 끝까지 두고 (점수, 수 목록)."""
    b = np.ascontiguousarray(np.asarray(board, np.int8).reshape(-1)).copy()
    buf = np.empty((MAXM, 5), np.int32)
    score, moves = 0, []
    while True:
        m = legal_moves(b, buf)
        if m == 0:
            return score, moves
        i = int(np.argmin(buf[:m, 4]))
        score += int(buf[i, 4])
        moves.append(tuple(int(x) for x in buf[i, :4]))
        apply_move(b, buf[i])
