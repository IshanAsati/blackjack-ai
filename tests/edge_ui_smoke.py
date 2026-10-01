"""Exercise the real finite-shoe API, autoplay, saved training curves, and layouts."""

from pathlib import Path
import sys

from playwright.sync_api import expect, sync_playwright

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
output = Path("runs/edge-ui-check")
output.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.goto(base)
    page.wait_for_load_state("networkidle")
    expect(page.get_by_role("heading", name="Let the model play.")).to_be_visible()
    expect(page.get_by_role("button", name="Start auto")).to_be_enabled()
    page.locator("#seed").fill("20261001")
    page.get_by_role("button", name="Start new session").click()
    expect(page.get_by_role("button", name="Next round", exact=True)).to_be_enabled()
    page.get_by_role("button", name="Next round", exact=True).click()
    expect(page.locator("#round-count")).to_have_text("1")
    expect(page.locator("#player-cards .card")).not_to_have_count(0)
    page.get_by_role("button", name="Find next bet").click()
    expect(page.locator("#bet-count")).to_have_text("1", timeout=30000)
    page.locator("#speed").select_option("1000")
    page.get_by_role("button", name="Start auto").click()
    page.wait_for_function("Number(document.getElementById('round-count').textContent.replaceAll(',', '')) > 1000")
    page.get_by_role("button", name="Pause auto").click()
    expect(page.get_by_role("button", name="Start auto")).to_be_enabled()
    count = page.locator("#round-count").inner_text()
    page.wait_for_timeout(600)
    expect(page.locator("#round-count")).to_have_text(count)
    page.screenshot(path=output / "live-desktop.png", full_page=True)
    page.reload()
    expect(page.locator("#round-count")).to_have_text(count)
    with page.expect_download() as download:
        page.get_by_role("button", name="Download session").click()
    download.value.save_as(output / "session.json")

    page.get_by_role("button", name="Training", exact=True).click()
    expect(page.locator("#training-chart polyline")).to_have_count(2)
    page.locator("#checkpoint").fill("0")
    expect(page.locator("#checkpoint-label")).to_have_text("25,000 episodes")
    expect(page.locator("#training-cards")).to_contain_text("Unsupervised representation")
    page.screenshot(path=output / "training-desktop.png", full_page=True)
    page.get_by_role("button", name="Live table", exact=True).click()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=output / "live-mobile.png", full_page=True)
    page.get_by_role("button", name="Training", exact=True).click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=output / "training-mobile.png", full_page=True)
    page.get_by_role("button", name="Live table", exact=True).click()
    page.set_viewport_size({"width": 1440, "height": 1080})
    page.locator("#seed").fill("20261001")
    page.get_by_role("button", name="Start new session").click()
    expect(page.locator("#round-count")).to_have_text("0")
    page.get_by_role("button", name="Run 100,000 rounds", exact=True).click()
    expect(page.locator("#round-count")).to_have_text("100,000", timeout=90000)
    expect(page.get_by_role("button", name="Start auto")).to_be_enabled()
    expect(page.locator("#control-note")).to_contain_text("Fixed run complete")
    page.screenshot(path=output / "fixed-run.png", full_page=True)
    with page.expect_download() as fixed_download:
        page.get_by_role("button", name="Download session").click()
    fixed_download.value.save_as(output / "fixed-run.json")
    assert page.request.post(base + "/api/edge/advance", data={"rounds": -1}).status == 400
    browser.close()
    assert not errors, errors
print("Live API, autoplay/pause, session restore/export, training scrubber, desktop and mobile passed.")
