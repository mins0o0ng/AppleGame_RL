"""게임 화면 -> 10x17 숫자 배열 (템플릿 매칭).

범용 OCR 없이:
1) 빨간 사과 픽셀의 행/열 투영으로 격자(칸 중심과 크기)를 찾고 (판이 꽉 찬 시작 화면에서 1회)
2) 칸마다 사과 안쪽 원 영역의 흰 픽셀(숫자)을 잘라 크기를 정규화한 뒤
3) 미리 만든 숫자 1~9 템플릿과 가장 가까운 것을 고른다. 빨간색이 없으면 빈 칸(0).
"""
import os

import numpy as np
from PIL import Image

from .env import ROWS, COLS

FEAT_W, FEAT_H = 10, 14
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "digit_templates.npy")


def load_rgb(img):
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    return img.astype(np.int16)


def red_mask(a):
    return (a[..., 0] > 150) & (a[..., 1] < 120) & (a[..., 2] < 120)


def _runs(profile, thresh):
    on = profile > thresh
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(on) - 1))
    return runs


def detect_grid(img):
    """사과가 꽉 찬 화면에서 격자를 찾는다. 반환: (열 중심 17개, 행 중심 10개, 칸 크기)."""
    red = red_mask(load_rgb(img))
    cols = _runs(red.sum(0), red.shape[0] * 0.04)
    rows = _runs(red.sum(1), red.shape[1] * 0.025)
    if len(cols) != COLS or len(rows) != ROWS:
        raise ValueError(f"격자 인식 실패: 열 {len(cols)}개, 행 {len(rows)}개 (기대 {COLS}x{ROWS}). "
                         "사과가 모두 있는 시작 화면의 게임판 영역만 캡처했는지 확인하세요.")
    cx = np.array([(s + e) / 2 for s, e in cols])
    cy = np.array([(s + e) / 2 for s, e in rows])
    size = float(np.mean([e - s + 1 for s, e in cols + rows]))
    return cx, cy, size


def _cell(a, x, y, size):
    h = int(size / 2)
    return a[max(0, int(y) - h):int(y) + h + 1, max(0, int(x) - h):int(x) + h + 1]


def cell_feature(crop):
    """칸 이미지 -> (사과 있음 여부, 정규화된 숫자 특징 벡터)."""
    h, w = crop.shape[:2]
    if red_mask(crop).mean() < 0.15:
        return False, None
    yy, xx = np.mgrid[:h, :w]
    inside = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (0.38 * min(h, w)) ** 2
    white = (crop.min(2) > 190) & inside
    ys, xs = np.nonzero(white)
    if len(ys) < 5:
        return True, np.zeros(FEAT_W * FEAT_H, np.float32)
    digit = white[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((digit * 255).astype(np.uint8)).resize((FEAT_W, FEAT_H), Image.BILINEAR)
    return True, np.asarray(im, np.float32).ravel() / 255.0


def board_features(img, grid):
    a = load_rgb(img)
    cx, cy, size = grid
    present = np.zeros((ROWS, COLS), bool)
    feats = np.zeros((ROWS, COLS, FEAT_W * FEAT_H), np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            ok, f = cell_feature(_cell(a, cx[c], cy[r], size))
            present[r, c] = ok
            if ok:
                feats[r, c] = f
    return present, feats


def build_templates(img, labels, path=TEMPLATE_PATH):
    """라벨이 있는 화면으로 숫자 1~9 평균 템플릿을 만들어 저장."""
    grid = detect_grid(img)
    present, feats = board_features(img, grid)
    labels = np.asarray(labels)
    tmpl = np.stack([feats[(labels == d) & present].mean(0) for d in range(1, 10)])
    np.save(path, tmpl)
    return tmpl


def read_board(img, grid=None, templates=None):
    """화면 -> (보드 (10,17) int8, 칸별 신뢰도 마진). 마진이 작으면 오인식 의심."""
    if grid is None:
        grid = detect_grid(img)
    if templates is None:
        templates = np.load(TEMPLATE_PATH)
    present, feats = board_features(img, grid)
    d = ((feats[..., None, :] - templates[None, None]) ** 2).sum(-1)   # (R, C, 9)
    order = np.sort(d, -1)
    board = (d.argmin(-1) + 1) * present
    margin = np.where(present, order[..., 1] - order[..., 0], np.inf)
    return board.astype(np.int8), margin


def cell_center(grid, r, c):
    cx, cy, _ = grid
    return float(cx[c]), float(cy[r])
