"""Browser smoke test run through the webapp-testing helper."""

from pathlib import Path
import sys
from playwright.sync_api import sync_playwright


output = Path("runs/ui-check")
output.mkdir(parents=True, exist_ok=True)
errors = []
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 950}, device_scale_factor=1)
    page.on("console", lambda message: errors.append(f"console: {message.text}") if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(f"page: {error}"))
    page.goto(base + "/classic")
    page.wait_for_load_state("networkidle")
    assert page.get_by_role("heading", name="Learn the decision, one hand at a time.").is_visible()
    assert page.get_by_role("button", name="Hit").is_disabled()

    page.get_by_role("button", name="Deal a hand").click()
    page.wait_for_selector("#player-cards .card")
    assert page.locator("#player-cards .card").count() == 2
    assert page.locator("#dealer-cards .card").count() == 2
    assert page.locator("#advice-action").inner_text().lower() in {"hit", "stand"}
    assert page.get_by_role("button", name="Hit").is_enabled()
    page.screenshot(path=output / "desktop.png", full_page=True)

    advice = page.locator("#advice-action").inner_text()
    page.get_by_role("button", name=advice, exact=True).click()
    page.wait_for_timeout(150)
    assert page.locator("#agreement").is_visible()
    if page.get_by_role("button", name="Stand", exact=True).is_enabled():
        page.get_by_role("button", name="Stand", exact=True).click()
    page.wait_for_selector("#status-label:text-is('You won'), #status-label:text-is('Dealer wins'), #status-label:text-is('Push')")
    assert int(page.locator("#wins").inner_text()) + int(page.locator("#draws").inner_text()) + int(page.locator("#losses").inner_text()) == 1

    mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    mobile.goto(base + "/classic")
    mobile.wait_for_load_state("networkidle")
    mobile.get_by_role("button", name="Deal a hand").click()
    mobile.wait_for_selector("#player-cards .card")
    mobile.screenshot(path=output / "mobile.png", full_page=True)
    assert mobile.locator("body").evaluate("el => el.scrollWidth <= el.clientWidth")
    browser.close()

assert not errors, errors
print(f"UI smoke test passed; screenshots are in {output}")
