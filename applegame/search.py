"""학습된 가치망/정책망을 이용한 빔 서치.

각 단계에서 빔의 모든 상태에 대해 정책 상위 k개 수를 펼치고,
(지금까지 점수 + 가치망이 예측한 남은 점수)가 높은 W개 상태만 남긴다.
"""
import torch

from .env import ROWS, COLS


@torch.no_grad()
def beam_search(model, board, width=16, topk=8, value_scale=10.0):
    rects = model.rects
    beam = board[None].clone()
    acc = torch.zeros(1, device=board.device)
    best = 0.0
    while len(beam):
        mask, n = rects.legal(beam)
        alive = mask.any(1)
        if (~alive).any():
            best = max(best, acc[~alive].max().item())
        beam, acc, mask, n = beam[alive], acc[alive], mask[alive], n[alive]
        if not len(beam):
            break
        logits, _ = model(beam, mask, n)
        k = min(topk, mask.shape[1])
        top = logits.topk(k, 1).indices
        valid = mask.gather(1, top)
        parent = torch.arange(len(beam), device=board.device)[:, None].expand(-1, k)[valid]
        acts = top[valid]
        child = beam[parent].masked_fill(rects.cover[acts].view(-1, ROWS, COLS), 0)
        child_acc = acc[parent] + n[parent, acts].float()
        # 동일한 보드 중복 제거
        uniq, inv = torch.unique(child.flatten(1), dim=0, return_inverse=True)
        first = torch.full((len(uniq),), len(inv), device=board.device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(inv), device=board.device), "amin")
        child, child_acc = child[first], child_acc[first]
        cm, cn = rects.legal(child)
        _, v = model(child, cm, cn)
        v = torch.where(cm.any(1), v * value_scale, torch.zeros_like(v))
        order = (child_acc + v).argsort(descending=True)[:width]
        beam, acc = child[order], child_acc[order]
    return best


@torch.no_grad()
def rollout_search(board, rects, rollout_policy, gen=None):
    """1-수 앞보기 롤아웃: 모든 유효 수를 두어 본 뒤 각각 rollout_policy로 끝까지 진행해
    (즉시 점수 + 롤아웃 점수)가 가장 큰 수를 고른다. 롤아웃 정책보다 나빠지지 않는다."""
    from .policies import evaluate
    board = board.clone()
    score = 0
    while True:
        mask, n = rects.legal(board[None])
        acts = mask[0].nonzero().squeeze(1)
        if not len(acts):
            return score
        child = board[None].expand(len(acts), -1, -1).masked_fill(rects.cover[acts].view(-1, ROWS, COLS), 0)
        total = n[0, acts] + evaluate(rollout_policy, child, rects)
        a = acts[total.argmax()]
        score += n[0, a].item()
        board = board.masked_fill(rects.cover[a].view(ROWS, COLS), 0)
