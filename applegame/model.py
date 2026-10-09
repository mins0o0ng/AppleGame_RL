"""CNN 몸통 + '직사각형 점수화' 정책 헤드 + 가치 헤드.

8415개 행동마다 Linear 출력을 두는 대신, 셀 단위 점수 맵을 만든 뒤
직사각형 내부 합 + 좌상/우하 모서리 점수 + 지울 사과 수의 함수로 로짓을 만든다.
(모두 셀->행동 행렬곱 한 번으로 계산)
파라미터 수가 행동 수와 무관하고, 보드의 위치 구조를 그대로 활용한다.
"""
import torch
import torch.nn as nn

from .env import ROWS, COLS, encode


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class AppleNet(nn.Module):
    def __init__(self, rects, channels=64, blocks=4, greedy_prior=0.0):
        super().__init__()
        self.rects = rects
        # 입력: 원-핫 10채널 + 행/열 좌표 2채널
        self.stem = nn.Sequential(nn.Conv2d(12, channels, 3, padding=1), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.cell_head = nn.Conv2d(channels, 3, 1)  # 내부합 / 좌상 / 우하 점수 맵
        # 지울 사과 수 n 에 대한 (n, n^2, 1/n) 가중치.
        # greedy_prior > 0 이면 '사과를 적게 지우는 수' 선호(탐욕 기준선)에서 출발한다.
        self.n_weight = nn.Parameter(torch.tensor([-greedy_prior, 0.0, 0.0]))
        self.value = nn.Sequential(nn.Flatten(), nn.Linear(channels * 2, 128), nn.ReLU(), nn.Linear(128, 1))
        ys = torch.linspace(-1, 1, ROWS).view(1, 1, ROWS, 1).expand(1, 1, ROWS, COLS)
        xs = torch.linspace(-1, 1, COLS).view(1, 1, 1, COLS).expand(1, 1, ROWS, COLS)
        self.register_buffer("coords", torch.cat([ys, xs], 1))

    def forward(self, boards, mask, n_removed):
        x = encode(boards)
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        h = self.body(self.stem(x))
        logits = self.cell_head(h).flatten(1) @ self.rects.head_t
        n = n_removed.float().clamp(min=1) / 10
        w = self.n_weight
        logits = logits + w[0] * n + w[1] * n * n + w[2] / n
        logits = logits.masked_fill(~mask, -1e9)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return logits, self.value(pooled).squeeze(-1)
