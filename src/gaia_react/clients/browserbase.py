"""Browserbase API client (minimal)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List


class BrowserbaseError(RuntimeError):
    pass


@dataclass
class BrowserbaseSession:
    id: str
    connect_url: str
    selenium_remote_url: str
    signing_key: str
    region: str
    status: str

    @property
    def replay_url(self) -> str:
        return f"https://browserbase.com/sessions/{self.id}"


class BrowserbaseClient:
    BASE_URL = "https://api.browserbase.com"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("BROWSERBASE_API_KEY")
        if not self.api_key:
            raise ValueError("BROWSERBASE_API_KEY not set")
        self.timeout = int(timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError("httpx package required. Install with: pip install httpx") from e
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"X-BB-API-Key": self.api_key},
                timeout=self.timeout,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def create_session(
        self,
        *,
        project_id: Optional[str] = None,
        proxies: Union[bool, List[Dict[str, Any]]] = False,
        # advanced_stealth may fall back if unavailable for the account.
        advanced_stealth: bool = False,
        browser_settings: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        keep_alive: Optional[bool] = None,
        region: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> BrowserbaseSession:
        """
        Create a Browserbase session.
        """
        project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        if not project_id:
            raise ValueError("BROWSERBASE_PROJECT_ID not set")

        base_settings: Dict[str, Any] = {}
        if browser_settings:
            base_settings.update(browser_settings)
        if advanced_stealth:
            base_settings.setdefault("advancedStealth", True)
        base_settings.setdefault("solveCaptchas", True)
        base_settings.setdefault("recordSession", True)
        base_settings.setdefault("logSession", True)

        payload: Dict[str, Any] = {
            "projectId": project_id,
            "proxies": proxies,
            "browserSettings": base_settings,
            "userMetadata": user_metadata or {"source": "gaia-react-data-collector"},
        }
        if timeout_seconds is not None:
            payload["timeout"] = int(timeout_seconds)
        if keep_alive is not None:
            payload["keepAlive"] = bool(keep_alive)
        if region is not None:
            payload["region"] = str(region)

        async def do_create(p: Dict[str, Any]) -> BrowserbaseSession:
            client = self._get_client()
            resp = await client.post("/v1/sessions", json=p)
            if resp.status_code != 201:
                raise BrowserbaseError(f"Browserbase create session failed: {resp.status_code} {resp.text}")
            data = resp.json()
            return BrowserbaseSession(
                id=str(data["id"]),
                connect_url=str(data["connectUrl"]),
                selenium_remote_url=str(data.get("seleniumRemoteUrl") or ""),
                signing_key=str(data.get("signingKey") or ""),
                region=str(data.get("region") or ""),
                status=str(data.get("status") or ""),
            )

        try:
            return await do_create(payload)
        except BrowserbaseError:
            if not advanced_stealth:
                raise
            payload2 = dict(payload)
            bs = dict(payload2.get("browserSettings") or {})
            bs.pop("advancedStealth", None)
            payload2["browserSettings"] = bs
            return await do_create(payload2)

    async def get_downloads_zip(self, session_id: str) -> bytes:
        """
        Retrieve downloads as a ZIP archive.
        """
        client = self._get_client()
        resp = await client.get(f"/v1/sessions/{session_id}/downloads")
        if resp.status_code != 200:
            raise BrowserbaseError(f"Browserbase get downloads failed: {resp.status_code} {resp.text}")
        return bytes(resp.content)

    async def delete_downloads(self, session_id: str) -> None:
        client = self._get_client()
        resp = await client.delete(f"/v1/sessions/{session_id}/downloads")
        if resp.status_code not in (204, 404):
            raise BrowserbaseError(f"Browserbase delete downloads failed: {resp.status_code} {resp.text}")

