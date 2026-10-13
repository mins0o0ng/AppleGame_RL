"""기준선 정책과 평가 루프."""
import torch

from .env import ROWS, COLS, RectIndex, sample_boards


def random_policy(boards, mask, n, gen=None):
    noise = torch.rand(mask.shape, generator=gen, device=mask.device)
    return noise.masked_fill(~mask, -1).argmax(1)


def greedy_min_policy(boards, mask, n, gen=None):
    """유효한 수 중 사과를 가장 적게 지우는 수 (동점은 무작위)."""
    noise = torch.rand(mask.shape, generator=gen, device=mask.device) * 0.1
    score = (-n.float() + noise).masked_fill(~mask, -1e9)
    return score.argmax(1)


def make_smart_policy(rects, w_kill=0.1, w_big=0.3):
    """탐욕 개선판: 사과를 적게 지우는 수를 우선하되,
    - w_kill: 이 수의 사과들이 다른 유효 수에 많이 쓰일수록(겹치는 수가 많을수록) 감점 -> 다른 기회를 덜 망가뜨리는 수 선호
    - w_big: 8, 9가 들어간 수 가점 (짝이 1, 2뿐이라 남기면 고립되기 쉬움)
    실제 보드 6장 단독 플레이: 탐욕 117.5 -> 122.5"""
    cover = rects.cover.float()
    cover_t = cover.T.contiguous()

    def pol(boards, mask, n, gen=None):
        flat = boards.flatten(1)
        deg = (mask.float() @ cover) * (flat > 0).float()
        pri = -n.float() - w_kill * (deg @ cover_t) + w_big * ((flat >= 8).float() @ cover_t)
        noise = torch.rand(mask.shape, generator=gen, device=mask.device) * 0.01
        return (pri + noise).masked_fill(~mask, -1e9).argmax(1)
    return pol


def model_policy(model, sample=False):
    @torch.no_grad()
    def act(boards, mask, n, gen=None):
        logits, _ = model(boards, mask, n)
        if sample:
            return torch.distributions.Categorical(logits=logits).sample()
        return logits.argmax(1)
    return act


@torch.no_grad()
def evaluate(policy, boards, rects=None, seed=0):
    """고정 보드 묶음 (N, rows, cols)을 끝까지 플레이하고 판별 점수 (N,)를 반환."""
    rects = rects or RectIndex(device=boards.device)
    gen = torch.Generator(device=boards.device).manual_seed(seed)
    boards = boards.clone()
    score = torch.zeros(len(boards), dtype=torch.long, device=boards.device)
    alive = torch.arange(len(boards), device=boards.device)
    while len(alive):
        b = boards[alive]
        mask, n = rects.legal(b)
        has = mask.any(1)
        alive, b, mask, n = alive[has], b[has], mask[has], n[has]
        if not len(alive):
            break
        a = policy(b, mask, n, gen)
        score[alive] += n.gather(1, a[:, None]).squeeze(1)
        boards[alive] = b.masked_fill(rects.cover[a].view(-1, ROWS, COLS), 0)
    return score


def eval_boards(num=1000, seed=12345, device="cpu", dist="uniform"):
    # CPU에서 생성해 옮긴다: GPU/CPU 어디서 돌려도 같은 평가 보드가 나오도록
    gen = torch.Generator().manual_seed(seed)
    return sample_boards(num, gen, dist).to(device)
