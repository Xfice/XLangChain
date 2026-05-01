"""Minimal, opt-in Playwright source for public-page demo scraping."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus


def fetch_public_x_with_playwright(keyword: str, limit: int) -> list[dict[str, Any]]:
    """Fetch public post-like snippets from X search page for demo use.

    Safety constraints:
    - Disabled unless PLAYWRIGHT_DEMO_ENABLED=true
    - Limit capped to 20
    - Short timeout and no login flows
    - Public page only (no private/account endpoints)
    """
    enabled = os.getenv("PLAYWRIGHT_DEMO_ENABLED", "false").lower() == "true"
    if not enabled:
        raise ValueError(
            "Playwright source is disabled. Set PLAYWRIGHT_DEMO_ENABLED=true to enable demo scraping."
        )

    if limit > 20:
        limit = 20

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import behavior depends on runtime
        raise ValueError(
            "Playwright is not installed. Install with `pip install .[scrape]` and run "
            "`python -m playwright install chromium`."
        ) from exc

    search_url = (
        "https://x.com/search?q="
        f"{quote_plus(keyword)}%20-is%3Aretweet%20lang%3Aen&src=typed_query&f=live"
    )
    items: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=12000)
            page.wait_for_timeout(1500)
            elements = page.locator("article div[lang]").all()
            for element in elements:
                text = element.inner_text().strip()
                if text:
                    items.append(
                        {
                            "text": text,
                            "sentiment": "unknown",
                            "date": datetime.now(tz=timezone.utc).isoformat(),
                        }
                    )
                if len(items) >= limit:
                    break
        except PlaywrightTimeoutError:
            pass
        finally:
            context.close()
            browser.close()

    return items
