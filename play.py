"""실제 게임 자동 플레이 (Windows/Mac, 브라우저 게임 화면을 그대로 읽고 마우스로 드래그).

사용법 (본인 PC에서):
    pip install -r requirements.txt mss pyautogui
    1) 브라우저에서 게임을 열고 '시작'을 눌러 사과판이 보이게 한다 (창을 가리지 말 것)
    2) python play.py --dry-run      # 마우스는 안 움직이고 인식/계획만 확인 + 인식 결과 이미지 저장
    3) python play.py                # 실제 플레이 (3초 카운트다운 뒤 시작)

멈추기: 마우스를 화면 왼쪽 위 모서리로 세게 옮기면 pyautogui 안전장치가 작동해 즉시 멈춘다. 또는 Ctrl+C.

동작: 매 수마다 화면을 캡처 -> 보드 인식 -> 롤아웃 탐색으로 한 수 선택 -> 드래그 -> 다시 캡처해 반영 확인.
드래그가 반영되지 않으면 선택 상자를 조금 넓혀 한 번 더 시도한다.
"""
import argparse
import time

import numpy as np
import torch

from applegame.env import RectIndex
from applegame.search import rollout_best_move
from applegame.vision import locate_grid, read_board


class Screen:
    """실제 화면 캡처 + 마우스 (mss, pyautogui)."""

    def __init__(self, monitor=1, drag_time=0.12):
        import sys
        if sys.platform == "win32":
            # Windows 화면 배율(125%, 150%...) 때문에 캡처 좌표와 마우스 좌표가 어긋나지 않게
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        import mss
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self.pg = pyautogui
        self.sct = mss.mss()
        self.mon = self.sct.monitors[monitor]
        self.drag_time = drag_time

    def capture(self):
        shot = self.sct.grab(self.mon)
        return np.asarray(shot)[..., [2, 1, 0]]  # BGRA -> RGB

    def drag(self, x0, y0, x1, y1, steps=10):
        """실제 마우스 드래그. 브라우저가 드래그로 인식하도록 누른 뒤 잠깐 멈추고 여러 단계로 나눠 움직인다."""
        ox, oy = self.mon["left"], self.mon["top"]
        self.pg.moveTo(x0 + ox, y0 + oy)
        time.sleep(0.03)
        self.pg.mouseDown()
        time.sleep(0.05)
        for i in range(1, steps + 1):
            t = i / steps
            self.pg.moveTo(x0 + (x1 - x0) * t + ox, y0 + (y1 - y0) * t + oy)
            time.sleep(self.drag_time / steps)
        time.sleep(0.05)
        self.pg.mouseUp()


def drag_box(grid, move, expand=0.42):
    """(r1, c1, r2, c2) -> 화면 좌표 드래그 시작/끝. 칸 간격의 expand배만큼 바깥으로."""
    cx, cy, _ = grid
    r1, c1, r2, c2 = move
    px = (cx[-1] - cx[0]) / (len(cx) - 1)
    py = (cy[-1] - cy[0]) / (len(cy) - 1)
    return (cx[c1] - expand * px, cy[r1] - expand * py, cx[c2] + expand * px, cy[r2] + expand * py)


def _stable_read(screen, grid, settle, max_wait=1.5):
    """드래그 후 사라지는 효과/애니메이션이 끝날 때까지 기다린다: 연속 두 번 같은 보드가 읽히면 확정."""
    t0 = time.time()
    time.sleep(settle)
    img = screen.capture()
    board, margin = read_board(img, grid)
    while time.time() - t0 < max_wait:
        time.sleep(max(settle, 0.05))
        img2 = screen.capture()
        board2, margin2 = read_board(img2, grid)
        if (board2 == board).all():
            return img2, board2, margin2
        img, board, margin = img2, board2, margin2
    return img, board, margin


def _save_debug(before, after, grid, move, tag):
    from PIL import Image, ImageDraw
    for name, im in (("before", before), ("after", after)):
        pic = Image.fromarray(np.ascontiguousarray(im))
        ImageDraw.Draw(pic).rectangle(drag_box(grid, move), outline=(0, 0, 255), width=3)
        pic.save(f"debug_{tag}_{name}.png")


def play(screen, time_limit=118.0, dry_run=False, settle=0.15, log=print):
    rects = RectIndex()
    img = screen.capture()
    try:
        grid = locate_grid(img)
    except ValueError:
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(img)).save("capture_fail.png")
        log("게임판을 찾지 못했습니다. 캡처한 화면을 capture_fail.png로 저장했습니다 (이 파일을 보내 주세요).")
        raise
    log(f"격자 찾음: 칸 크기 {grid[2]:.0f}px, 좌상단 칸 중심 ({grid[0][0]:.0f}, {grid[1][0]:.0f})")
    start, score, moves, fails = time.time(), 0, 0, 0
    board, margin = read_board(img, grid)
    rec = open("play_log.txt", "w", encoding="utf-8")
    rec.write("start\n" + "\n".join(" ".join(map(str, r)) for r in board) + "\n")
    while time.time() - start < time_limit:
        weak = int((margin < 1.0).sum())
        if weak:
            log(f"  !! 인식 불확실 칸 {weak}개")
        pick = rollout_best_move(torch.tensor(board), rects)
        if pick is None:
            log("더 둘 수 있는 수가 없습니다.")
            break
        move, n, expect = pick
        if dry_run:
            log(f"[dry-run] 다음 수 {move} (사과 {n}개, 이후 예상 +{expect})")
            return board, grid, move
        r1, c1, r2, c2 = move
        expected = board.copy()
        expected[r1:r2 + 1, c1:c2 + 1] = 0
        before = img
        for expand in (0.42, 0.48):
            screen.drag(*drag_box(grid, move, expand))
            img, new, margin = _stable_read(screen, grid, settle)
            if (new != board).any():
                break
        if (new == board).all():
            fails += 1
            log(f"  !! 드래그가 반영되지 않음 {move} (누적 {fails})")
            if fails >= 3:
                _save_debug(before, img, grid, move, "nochange")
                log("연속 실패: 디버그 이미지(debug_nochange_*.png)를 저장하고 중단합니다.")
                break
            continue
        gained = n
        if (new != expected).any():
            # 계획한 사과 말고 다른 칸도 바뀜. 콤보/점수 팝업처럼 잠깐 가리는 효과일 수 있으니 2초까지 다시 읽어 본다
            bad = int((new != expected).sum())
            t0 = time.time()
            while (new != expected).any() and time.time() - t0 < 2.0:
                time.sleep(0.2)
                img = screen.capture()
                try:
                    new, margin = read_board(img, grid)
                except ValueError:
                    continue
            if (new != expected).any():
                try:   # 게임판 위치가 바뀌었을 수 있음
                    g2 = locate_grid(img)
                    n2, m2 = read_board(img, g2)
                    if not (n2 != expected).any():
                        grid, new, margin = g2, n2, m2
                        log("  (격자 위치가 바뀌어 다시 찾음)")
                except ValueError:
                    pass
            if (new != expected).any():
                only_removed = ((new == board) | (new == 0)).all()
                extra = int(((expected > 0) & (new == 0)).sum())
                if only_removed and extra <= 6 and (margin[new > 0] >= 1.0).all():
                    # 사과가 줄어들기만 함 -> 새로 읽은 화면을 믿고 계속
                    gained = int((board > 0).sum() - (new > 0).sum())
                    _save_debug(before, img, grid, move, f"resync{moves + 1}")
                    log(f"  !! 예상과 {bad}칸 다름 -> 화면을 새로 읽은 보드로 계속 (지운 사과 {gained}개)")
                elif (new[r1:r2 + 1, c1:c2 + 1] == 0).all():
                    # 계획한 칸은 모두 비었고 화면이 크게 바뀜 = 마지막 수를 둔 직후 게임 종료 창이 뜬 경우
                    score += n
                    moves += 1
                    log(f"{moves:3d}수 {move} +{n} = {score}점 ({time.time() - start:5.1f}초)")
                    rec.write(f"move {moves} {move} +{n} (마지막 수, 게임 종료 화면)\n")
                    log("게임 종료 화면을 확인했습니다.")
                    break
                else:
                    _save_debug(before, img, grid, move, "unexpected")
                    log(f"  !! 큰 화면 변화 ({bad}칸): 게임 종료 화면이거나 화면이 바뀐 것으로 보고 멈춥니다. "
                        "debug_unexpected_before.png / after.png와 play_log.txt를 보내 주세요.")
                    break
        fails = 0
        score += gained
        moves += 1
        log(f"{moves:3d}수 {move} +{gained} = {score}점 ({time.time() - start:5.1f}초)")
        rec.write(f"move {moves} {move} +{gained}\n" + "\n".join(" ".join(map(str, r)) for r in new) + "\n")
        rec.flush()
        board = new
    log(f"끝: {score}점, {moves}수, {time.time() - start:.1f}초 (수 기록: play_log.txt)")
    rec.write(f"end {score} {moves}\n")
    rec.close()
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="마우스를 움직이지 않고 인식과 첫 수만 확인")
    p.add_argument("--monitor", type=int, default=1, help="캡처할 모니터 번호 (1 = 주 모니터)")
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--time-limit", type=float, default=118.0)
    p.add_argument("--drag-time", type=float, default=0.12, help="드래그에 쓰는 시간(초). 인식이 안 되면 늘리기")
    args = p.parse_args()

    screen = Screen(args.monitor, args.drag_time)
    for i in range(int(args.countdown), 0, -1):
        print(f"{i}...", flush=True)
        time.sleep(1)
    result = play(screen, args.time_limit, args.dry_run)
    if args.dry_run:
        board, grid, move = result
        from PIL import Image, ImageDraw
        img = Image.fromarray(screen.capture())
        d = ImageDraw.Draw(img)
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                d.text((grid[0][c] - 4, grid[1][r] - 26), str(int(board[r, c])), fill=(0, 0, 255))
        d.rectangle(drag_box(grid, move), outline=(0, 0, 255), width=3)
        img.save("dry_run.png")
        print("인식 결과(파란 숫자)와 첫 수(파란 상자)를 dry_run.png에 저장했습니다.")
        print("인식한 보드:")
        for row in board:
            print(" ".join(str(int(v)) for v in row))


if __name__ == "__main__":
    main()
