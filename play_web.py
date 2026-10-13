"""브라우저 직접 제어 자동 플레이 (Playwright).

play.py(화면 캡처 + OS 마우스)와 달리, 웹페이지 안에서 캡처하고 마우스 입력을 보낸다.
- 좌표가 웹페이지(CSS 픽셀) 기준이라 Windows 화면 배율, 창 위치, 다른 창에 가려지는 문제가 없다
- 마우스 입력이 브라우저 내부의 실제 사용자 입력(isTrusted)으로 들어간다
- 게임 내부 코드나 점수는 건드리지 않는다 (사람이 마우스로 하는 것과 같은 입력만 보낸다)

설치 (한 번만):
    pip install playwright
    (설치된 Chrome을 쓰므로 브라우저를 따로 받을 필요 없음. Chrome이 없으면: python -m playwright install chromium)

사용:
    python play_web.py            # 새 Chrome 창에 메인 페이지가 열림 -> 그 창에서 게임 시작 -> 사과판이 보이면 터미널에서 Enter
    python play_web.py --dry-run  # 인식과 첫 수만 확인 (web_dry_run.png 저장)
    게임 링크(/play/...)는 일회용일 수 있으므로 붙여 넣지 말고, 열린 창에서 직접 게임을 시작한다.
"""
import argparse
import io

import numpy as np
from PIL import Image

from play import drag_box, play


class BrowserScreen:
    """Playwright 페이지를 play.play()가 쓰는 화면처럼 감싼다 (capture / drag)."""

    def __init__(self, page, drag_steps=8, hold=0.03):
        self.page = page
        self.drag_steps = drag_steps
        self.hold = hold

    def capture(self):
        # scale="css": 화면 배율과 상관없이 마우스 좌표와 같은 CSS 픽셀 단위로 캡처
        png = self.page.screenshot(scale="css")
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def drag(self, x0, y0, x1, y1):
        m = self.page.mouse
        m.move(x0, y0)
        m.down()
        self.page.wait_for_timeout(self.hold * 1000)
        m.move(x1, y1, steps=self.drag_steps)
        self.page.wait_for_timeout(self.hold * 1000)
        m.up()


def open_page(pw, url, browser="chrome", cdp=None, profile=None):
    if cdp:
        # 이미 열어 둔 Chrome에 붙기: chrome.exe --remote-debugging-port=9222 로 실행한 경우
        b = pw.chromium.connect_over_cdp(cdp)
        ctx = b.contexts[0]
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if url:
            page.goto(url)
        return b, page
    kw = {"headless": False, "args": ["--start-maximized"]}
    if browser == "chrome":
        kw["channel"] = "chrome"
    elif browser not in ("chromium", None):
        kw["executable_path"] = browser
    if profile:
        # 전용 프로필 폴더: 로그인/쿠키/사이트 설정이 다음 실행에도 남는다
        ctx = pw.chromium.launch_persistent_context(profile, no_viewport=True, **kw)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if url:
            page.goto(url)
        return ctx, page
    b = pw.chromium.launch(**kw)
    page = b.new_page(no_viewport=True)
    if url:
        page.goto(url)
    return b, page


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://apple.oshizi.com/ko/",
                   help="처음 열 주소. 게임 링크는 일회용일 수 있으니 메인 페이지에서 직접 게임을 시작하는 것을 권장")
    p.add_argument("--profile", default="chrome_profile",
                   help="전용 Chrome 프로필 폴더 (접속 정보 유지). 빈 문자열이면 매번 새 프로필")
    p.add_argument("--browser", default="chrome", help="chrome(설치된 Chrome) / chromium / 브라우저 실행 파일 경로")
    p.add_argument("--cdp", default=None, help="이미 열린 Chrome에 붙기 (예: http://localhost:9222)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--time-limit", type=float, default=115.0)
    p.add_argument("--settle", type=float, default=0.25, help="드래그 후 화면 반영을 기다리는 시간(초)")
    p.add_argument("--plan-time", type=float, default=-1, help="미리 계산 시간(초). -1 = 자동")
    p.add_argument("--move-sec", type=float, default=0.9)
    p.add_argument("--no-wait", action="store_true", help="Enter 입력 없이 바로 시작 (테스트용)")
    args = p.parse_args()

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser, page = open_page(pw, args.url, args.browser, args.cdp, args.profile or None)
        if not args.no_wait:
            input("브라우저에서 게임을 시작하고 사과판이 다 보이면 여기서 Enter를 누르세요...")
        screen = BrowserScreen(page)
        result = play(screen, args.time_limit, args.dry_run, settle=args.settle, plan_time=args.plan_time,
                      move_sec=args.move_sec)
        if args.dry_run:
            board, grid, move = result
            from PIL import ImageDraw
            img = Image.fromarray(screen.capture())
            d = ImageDraw.Draw(img)
            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    d.text((grid[0][c] - 4, grid[1][r] - 26), str(int(board[r, c])), fill=(0, 0, 255))
            d.rectangle(drag_box(grid, move), outline=(0, 0, 255), width=3)
            img.save("web_dry_run.png")
            print("인식 결과와 첫 수를 web_dry_run.png에 저장했습니다.")
        if not args.no_wait:
            input("끝났습니다. Enter를 누르면 브라우저를 닫습니다...")
        browser.close()


if __name__ == "__main__":
    main()
