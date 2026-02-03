"""
Browser tool: Execute Playwright scripts against a remote Browserbase session.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..clients.browserbase import BrowserbaseClient, BrowserbaseSession


@dataclass
class BrowserToolOutput:
    observation: str
    session_id: Optional[str] = None
    session_replay_url: Optional[str] = None
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    screenshot_data_url: Optional[str] = None  # data:image/...;base64,...
    downloaded_files: Optional[List[str]] = None


class BrowserbasePlaywrightTool:
    """
    Remote browser tool that executes Playwright scripts.
    """

    def __init__(
        self,
        *,
        browserbase_client: Optional[BrowserbaseClient] = None,
        project_id: Optional[str] = None,
        proxies: bool = False,
        advanced_stealth: bool = False,
        session_timeout_seconds: Optional[int] = 600,
        artifacts_dir: Optional[str] = None,
    ):
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        self.proxies = bool(proxies)
        self.advanced_stealth = bool(advanced_stealth)
        self.session_timeout_seconds = session_timeout_seconds
        self.browserbase = browserbase_client or BrowserbaseClient()

        self._session: Optional[BrowserbaseSession] = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._captcha_solving = False
        self._captcha_done_event = None

        self.artifacts_dir = artifacts_dir or tempfile.mkdtemp(prefix="bb_playwright_")
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Store last screenshot for multimodal
        self._last_screenshot_data_url: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session.id if self._session else None

    @property
    def session_replay_url(self) -> Optional[str]:
        return self._session.replay_url if self._session else None

    async def _ensure_connected(self) -> None:
        if self._page is not None:
            try:
                if not self._page.is_closed():
                    return
            except Exception:
                pass

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError("playwright package required. Install with: pip install playwright") from e

        if self._playwright is None:
            self._playwright = await async_playwright().start()

        if self._session is None:
            self._session = await self.browserbase.create_session(
                project_id=self.project_id,
                proxies=True if self.proxies else False,
                advanced_stealth=self.advanced_stealth,
                timeout_seconds=self.session_timeout_seconds,
                user_metadata={"source": "gaia-react-data-collector", "tool": "browserbase_playwright"},
            )

        if self._browser is None:
            self._browser = await self._playwright.chromium.connect_over_cdp(self._session.connect_url)

        contexts = list(getattr(self._browser, "contexts", []) or [])
        if contexts:
            self._context = contexts[0]
        else:
            self._context = await self._browser.new_context()

        pages = list(getattr(self._context, "pages", []) or [])
        if pages:
            self._page = pages[0]
        else:
            self._page = await self._context.new_page()

        # Enable downloads via CDP
        try:
            client = await self._context.new_cdp_session(self._page)
            await client.send(
                "Browser.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": "downloads", "eventsEnabled": True},
            )
        except Exception:
            pass

        # Listen for Browserbase captcha solver events
        try:
            self._captcha_done_event = asyncio.Event()

            def _handle_console(msg):
                try:
                    t = (msg.text or "").strip()
                except Exception:
                    t = ""
                if t == "browserbase-solving-started":
                    self._captcha_solving = True
                    try:
                        self._captcha_done_event.clear()
                    except Exception:
                        pass
                elif t == "browserbase-solving-finished":
                    self._captcha_solving = False
                    try:
                        self._captcha_done_event.set()
                    except Exception:
                        pass

            self._page.on("console", _handle_console)
        except Exception:
            pass

    async def _wait_for_captcha_solve(self, timeout_seconds: int = 30) -> None:
        if not self._captcha_solving:
            return
        try:
            if self._captcha_done_event is None:
                return
            await asyncio.wait_for(self._captcha_done_event.wait(), timeout=max(1, int(timeout_seconds)))
        except Exception:
            pass

    async def _maybe_restart_session_for_proxy(self, proxies_param: Optional[Any]) -> None:
        if proxies_param is None:
            return
        desired = proxies_param
        if isinstance(desired, str):
            desired = desired.strip().lower() in {"1", "true", "yes", "y"}
        desired = bool(desired)
        if desired == self.proxies:
            return
        await self.close()
        self.proxies = desired

    async def _screenshot_helper(self, full_page: bool = False, quality: int = 80) -> str:
        """Take screenshot and return base64 data URL."""
        img_bytes = await self._page.screenshot(type="jpeg", quality=quality, full_page=full_page)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        self._last_screenshot_data_url = data_url
        # Save locally too
        fn = os.path.join(self.artifacts_dir, f"screenshot_{self._session.id}_{int(time.time())}.jpeg")
        with open(fn, "wb") as f:
            f.write(img_bytes)
        return data_url

    async def _get_downloads_helper(self) -> List[str]:
        """List downloads in session ZIP."""
        assert self._session is not None
        data = await self.browserbase.get_downloads_zip(self._session.id)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
        return names

    async def _extract_downloads_helper(self, retry_seconds: int = 20) -> List[str]:
        """Extract downloads to local paths."""
        assert self._session is not None
        end = time.time() + max(1, int(retry_seconds))
        zip_bytes: Optional[bytes] = None
        while time.time() < end:
            try:
                data = await self.browserbase.get_downloads_zip(self._session.id)
                if len(data) > 64:
                    zip_bytes = data
                    break
            except Exception:
                pass
            await asyncio.sleep(2.0)

        if zip_bytes is None:
            raise RuntimeError(f"No downloads found after {retry_seconds}s")

        out_dir = os.path.join(self.artifacts_dir, f"downloads_{self._session.id}")
        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name or name.endswith("/"):
                    continue
                dest = os.path.join(out_dir, os.path.basename(name))
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                paths.append(dest)
        return paths

    async def execute(self, **params) -> BrowserToolOutput:
        script = str(params.get("script") or "").strip()
        if not script:
            return BrowserToolOutput(observation="Error: 'script' is required. Write Playwright code to execute.")

        # Allow model to enable proxies explicitly
        await self._maybe_restart_session_for_proxy(params.get("proxies"))
        await self._ensure_connected()

        # Reset last screenshot
        self._last_screenshot_data_url = None

        # Build execution scope with helpers
        scope: Dict[str, Any] = {
            "page": self._page,
            "context": self._context,
            "browser": self._browser,
            "screenshot": self._screenshot_helper,
            "get_downloads": self._get_downloads_helper,
            "extract_downloads": self._extract_downloads_helper,
            "asyncio": asyncio,
        }

        # Capture print output
        output_lines: List[str] = []

        def captured_print(*args, **kwargs):
            output_lines.append(" ".join(str(a) for a in args))

        scope["print"] = captured_print

        try:
            # Execute the script as async code
            wrapped = "async def __script__():\n"
            for line in script.split("\n"):
                wrapped += f"    {line}\n"
            wrapped += "\n__result__ = await __script__()"

            exec(compile(wrapped, "<browser_script>", "exec"), scope)
            result = await scope.get("__script__", lambda: None)()

            # Wait for any CAPTCHA solving
            await self._wait_for_captcha_solve(timeout_seconds=30)

            # Build observation
            obs_parts = []
            if output_lines:
                obs_parts.append("Output:\n" + "\n".join(output_lines))
            if result is not None:
                obs_parts.append(f"Return value: {result!r}")

            try:
                page_url = self._page.url
                page_title = await self._page.title()
            except Exception:
                page_url = None
                page_title = None

            obs_parts.append(f"\nSession: {self.session_id}")
            obs_parts.append(f"Replay: {self.session_replay_url}")
            if page_url:
                obs_parts.append(f"URL: {page_url}")
            if page_title:
                obs_parts.append(f"Title: {page_title}")

            return BrowserToolOutput(
                observation="OK\n" + "\n".join(obs_parts) if obs_parts else "OK: script executed.",
                session_id=self.session_id,
                session_replay_url=self.session_replay_url,
                page_url=page_url,
                page_title=page_title,
                screenshot_data_url=self._last_screenshot_data_url,
            )

        except Exception as e:
            tb = traceback.format_exc()
            try:
                page_url = self._page.url
                page_title = await self._page.title()
            except Exception:
                page_url = None
                page_title = None

            return BrowserToolOutput(
                observation=f"Error executing script:\n{e}\n\nTraceback:\n{tb}\n\nSession: {self.session_id}\nReplay: {self.session_replay_url}",
                session_id=self.session_id,
                session_replay_url=self.session_replay_url,
                page_url=page_url,
                page_title=page_title,
            )

    async def close(self) -> None:
        try:
            if self._page is not None and not self._page.is_closed():
                await self._page.close()
        except Exception:
            pass
        self._page = None
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:
            pass
        self._browser = None
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        try:
            await self.browserbase.aclose()
        except Exception:
            pass

