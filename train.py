"""마스킹 PPO 학습 스크립트.

예) python train.py --minutes 30
    python train.py --updates 500 --envs 256 --device cuda
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from applegame.env import BatchAppleEnv, RectIndex
from applegame.model import AppleNet
from applegame.policies import eval_boards, evaluate, model_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=64, help="환경당 롤아웃 길이")
    p.add_argument("--updates", type=int, default=10**9)
    p.add_argument("--minutes", type=float, default=30.0, help="학습 시간 제한")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.003)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--greedy-prior", type=float, default=20.0,
                   help="초기 정책을 탐욕(최소 제거) 쪽으로 기울이는 세기. 0이면 백지에서 시작")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-boards", type=int, default=200)
    p.add_argument("--dist", choices=["uniform", "real"], default="uniform",
                   help="보드 분포: uniform = 1~9 균등, real = 실제 게임 추정 분포(작은 숫자 많음, 합 10의 배수)")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"장치: {dev} | 모델 {args.channels}채널 x {args.blocks}블록", flush=True)
    if dev == "cpu":
        print("  !! GPU를 쓰지 않고 있습니다. CPU라면 --channels 32 --blocks 3 처럼 작은 모델을 권장합니다.", flush=True)
    os.makedirs(args.out, exist_ok=True)

    rects = RectIndex(device=dev)
    env = BatchAppleEnv(args.envs, device=dev, seed=args.seed, rects=rects, dist=args.dist)
    model = AppleNet(rects, args.channels, args.blocks, args.greedy_prior).to(dev)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_boards = eval_boards(args.eval_boards, seed=999, device=dev, dist=args.dist)  # 최종 평가용 보드와 시드가 다름

    T, N = args.steps, args.envs
    boards_buf = torch.zeros((T, N) + env.boards.shape[1:], dtype=torch.int8, device=dev)
    act_buf = torch.zeros(T, N, dtype=torch.long, device=dev)
    logp_buf = torch.zeros(T, N, device=dev)
    val_buf = torch.zeros(T, N, device=dev)
    rew_buf = torch.zeros(T, N, device=dev)
    done_buf = torch.zeros(T, N, device=dev)

    start = time.time()
    finished, best = [], -1.0
    for update in range(1, args.updates + 1):
        # ---- 롤아웃 수집 ----
        model.eval()
        with torch.no_grad():
            for t in range(T):
                boards_buf[t] = env.boards
                logits, value = model(env.boards, env.mask, env.n_removed)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                act_buf[t], logp_buf[t], val_buf[t] = a, dist.log_prob(a), value
                r, done, final = env.step(a)
                rew_buf[t], done_buf[t] = r / 10.0, done.float()
                finished += final[done].tolist()
            _, last_value = model(env.boards, env.mask, env.n_removed)

        # ---- GAE ----
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nxt = last_value if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * nxt * nonterm - val_buf[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        # ---- PPO 업데이트 ----
        model.train()
        b_boards = boards_buf.flatten(0, 1)
        b_act, b_logp, b_adv, b_ret = (x.flatten() for x in (act_buf, logp_buf, adv, ret))
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        total = T * N
        for _ in range(args.epochs):
            perm = torch.randperm(total, device=dev)
            for i in range(0, total, args.minibatch):
                idx = perm[i:i + args.minibatch]
                bb = b_boards[idx]
                mask, n = rects.legal(bb)
                logits, value = model(bb, mask, n)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                a_ = b_adv[idx]
                pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                vl = F.mse_loss(value, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + args.vf * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        elapsed = time.time() - start
        if finished:
            recent = finished[-500:]
            print(f"upd {update:4d} | {elapsed/60:5.1f}분 | 학습판 평균 {sum(recent)/len(recent):6.1f} "
                  f"| 엔트로피 {ent.item():.2f} | 가치손실 {vl.item():.3f} | {update*T*N/elapsed:.0f} 스텝/초",
                  flush=True)
            finished = finished[-500:]
        if update % args.eval_every == 0 or elapsed > args.minutes * 60:
            model.eval()
            score = evaluate(model_policy(model), val_boards, rects).float().mean().item()
            torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
            if score > best:
                best = score
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print(f"  >> 검증(탐욕적 행동) 평균 {score:.1f} | 최고 {best:.1f}", flush=True)
        if elapsed > args.minutes * 60:
            break


if __name__ == "__main__":
    main()
