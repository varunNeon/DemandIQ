"""
Keep the deployed Streamlit app awake.

Streamlit Community Cloud puts apps to sleep after a period without visitors.
This script opens the app in a headless browser, presses the wake-up button if
the app is asleep, and waits until DemandIQ has actually rendered.

Usage:
    python scripts/keep_awake.py https://your-app.streamlit.app
    STREAMLIT_APP_URL=https://your-app.streamlit.app python scripts/keep_awake.py

Requires: pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import os
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
APP_READY_TEXT = "Forecasts you can order stock against"


def wake(url: str, timeout_s: int = 240) -> bool:
    with sync_playwright() as p:
        # PLAYWRIGHT_CHANNEL=msedge reuses an installed Edge instead of a downloaded Chromium.
        browser = p.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL") or None)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        # The sleeping page shows a wake-up button; an awake app shows nothing like it.
        try:
            button = page.get_by_role("button", name=WAKE_BUTTON)
            button.wait_for(timeout=15_000)
            print("App is asleep. Waking it up.")
            button.click()
        except PlaywrightTimeout:
            print("No wake-up button found; the app looks awake.")

        # The app itself renders inside an iframe on *.streamlit.app.
        deadline_ms = timeout_s * 1000
        try:
            page.wait_for_function(
                """(text) => {
                    const docs = [document, ...Array.from(document.querySelectorAll('iframe'))
                        .map(f => { try { return f.contentDocument } catch (e) { return null } })
                        .filter(Boolean)];
                    return docs.some(d => d.body && d.body.innerText.includes(text));
                }""",
                arg=APP_READY_TEXT,
                timeout=deadline_ms,
            )
            ok = True
        except PlaywrightTimeout:
            ok = any(APP_READY_TEXT in (f.content() or "") for f in page.frames)
        browser.close()
    return ok


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("STREAMLIT_APP_URL", "")
    if not url:
        sys.exit("Pass the app URL as an argument or set STREAMLIT_APP_URL.")
    if wake(url):
        print(f"DemandIQ is up at {url}")
    else:
        sys.exit(f"DemandIQ did not finish loading at {url}")


if __name__ == "__main__":
    main()
