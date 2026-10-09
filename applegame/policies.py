"""기준선 정책과 평가 루프."""
import torch

from .env import ROWS, COLS, RectIndex


def random_policy(boards, mask, n, gen=None):
    noise = torch.rand(mask.shape, generator=gen, device=mask.device)
    return noise.masked_fill(~mask, -1).argmax(1)


def greedy_min_policy(boards, mask, n, gen=None):
    """유효한 수 중 사과를 가장 적게 지우는 수 (동점은 무작위)."""
    noise = torch.rand(mask.shape, generator=gen, device=mask.device) * 0.1
    score = (-n.float() + noise).masked_fill(~mask, -1e9)
    return score.argmax(1)


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


def eval_boards(num=1000, seed=12345, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(1, 10, (num, ROWS, COLS), generator=gen, device=device, dtype=torch.int8)
