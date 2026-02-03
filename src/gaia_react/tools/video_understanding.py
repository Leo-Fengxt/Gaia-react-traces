"""
Dedicated video understanding tool via OpenRouter.
"""

from __future__ import annotations

import base64
import io
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from ..clients.browserbase import BrowserbaseClient
from ..clients.openrouter import OpenRouterClient


SUPPORTED_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/mov",
    ".webm": "video/webm",
}


def _is_youtube_url(url: str) -> bool:
    u = (url or "").lower()
    return "youtube.com/" in u or "youtu.be/" in u


def _guess_mime_from_name(name: str) -> str:
    low = (name or "").lower()
    for ext, mt in SUPPORTED_VIDEO_MIME.items():
        if low.endswith(ext):
            return mt
    return "video/mp4"


@dataclass
class VideoUnderstandingResult:
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class VideoUnderstandingTool:
    """
    Dedicated tool to send video inputs to OpenRouter (Gemini video-capable model).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "google/gemini-3-flash-preview",
        max_tokens: int = 8192,
        timeout: int = 900,
    ):
        # Use a separate client pinned to the required model.
        self.llm = OpenRouterClient(
            api_key=api_key,
            model=model,
            temperature=0.2,
            top_p=0.95,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort="none",
            reasoning_exclude=True,
        )
        self.model = model

    async def _video_data_url_from_browserbase_downloads(
        self,
        *,
        session_id: str,
        preferred_filename: Optional[str] = None,
        retry_seconds: int = 20,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> str:
        bb = BrowserbaseClient()
        zip_bytes = await bb.get_downloads_zip(session_id)

        # If Browserbase hasn't synced yet, the zip can be tiny/empty. Retry at the tool level.
        if len(zip_bytes) < 64 and retry_seconds > 0:
            import asyncio, time

            end = time.time() + max(1, int(retry_seconds))
            while time.time() < end and len(zip_bytes) < 64:
                await asyncio.sleep(2)
                zip_bytes = await bb.get_downloads_zip(session_id)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
            if not names:
                raise RuntimeError("No downloaded files found in Browserbase session downloads ZIP.")

            chosen = names[0]
            if preferred_filename:
                pf = preferred_filename.lower().strip()
                for n in names:
                    if pf in n.lower():
                        chosen = n
                        break

            data = zf.read(chosen)

        if len(data) > int(max_bytes):
            raise RuntimeError(
                f"Downloaded video is too large ({len(data)} bytes). "
                f"Max supported by this tool is {int(max_bytes)} bytes; trim/compress the video first."
            )

        mime = _guess_mime_from_name(chosen)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    async def understand(
        self,
        *,
        prompt: str,
        video_url: Optional[str] = None,
        youtube_url: Optional[str] = None,
        video_data_url: Optional[str] = None,
        browserbase_session_id: Optional[str] = None,
        browserbase_filename: Optional[str] = None,
        browserbase_retry_seconds: int = 20,
    ) -> VideoUnderstandingResult:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        v: Optional[str] = None
        if youtube_url:
            v = str(youtube_url).strip()
        elif video_data_url:
            v = str(video_data_url).strip()
        elif browserbase_session_id:
            v = await self._video_data_url_from_browserbase_downloads(
                session_id=str(browserbase_session_id).strip(),
                preferred_filename=browserbase_filename,
                retry_seconds=int(browserbase_retry_seconds),
            )
        elif video_url:
            v = str(video_url).strip()

        if not v:
            raise ValueError("You must provide youtube_url, video_data_url, browserbase_session_id, or video_url.")

        # Enforce provider constraint: non-YouTube URLs generally won't be forwarded to Gemini.
        if v.startswith("http"):
            if not _is_youtube_url(v):
                raise ValueError(
                    "Non-YouTube video URLs are not supported for this tool/provider. "
                    "Download via Browserbase first, then call understand_video with browserbase_session_id "
                    "(or provide a base64 data:video/... URL)."
                )

        if v.startswith("data:") and "base64," not in v:
            raise ValueError("video_data_url must be a base64 data URL (data:video/...;base64,...)")

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": v}},
                ],
            }
        ]

        resp = await self.llm.achat(messages, stop=None)
        return VideoUnderstandingResult(content=resp.content or "", model=self.model, usage=resp.usage)

    async def execute(self, **params) -> str:
        res = await self.understand(
            prompt=params.get("prompt") or params.get("text") or "",
            video_url=params.get("video_url"),
            youtube_url=params.get("youtube_url"),
            video_data_url=params.get("video_data_url"),
            browserbase_session_id=params.get("browserbase_session_id"),
            browserbase_filename=params.get("browserbase_filename"),
            browserbase_retry_seconds=int(params.get("browserbase_retry_seconds", 20) or 20),
        )
        return res.content

