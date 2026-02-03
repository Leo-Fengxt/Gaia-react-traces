"""
Web Contents Tool using Exa API.
"""

from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass

from ..clients.exa import ExaClient


@dataclass
class PageContent:
    """Content from a single web page."""

    url: str
    title: str
    text: str
    published_date: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class WebContentsOutput:
    """Output from web contents tool."""

    pages: List[PageContent]
    error: Optional[str] = None
    request_id: Optional[str] = None
    cost_dollars: Optional[float] = None

    def to_observation(self) -> str:
        if self.error:
            return f"Web contents error: {self.error}"

        if not self.pages:
            return "No content retrieved."

        lines = []
        if self.request_id:
            lines.append(f"EXA requestId: {self.request_id}")
        if self.cost_dollars is not None:
            lines.append(f"EXA costDollars: {self.cost_dollars}")
        if lines:
            lines.append("")

        for page in self.pages:
            lines.append(f"=== {page.title} ===")
            lines.append(f"URL: {page.url}")

            if page.error:
                lines.append(f"ERROR: {page.error}")
            else:
                if page.published_date:
                    lines.append(f"Published: {page.published_date}")
                lines.append(f"Content ({len(page.text)} chars):")
                lines.append(page.text)

            lines.append("")
            lines.append("-" * 80)
            lines.append("")

        lines.append(
            "TIP: Extract the exact wording you need from the content above. If you've found the answer, use finish with \\boxed{...}."
        )
        return "\n".join(lines)


class WebContentsTool:
    """
    Web contents tool using Exa API (full text retrieval).
    """

    def __init__(
        self,
        exa_client: Optional[ExaClient] = None,
        max_characters: int = 50000,
        livecrawl: str = "fallback",
    ):
        self.exa_client = exa_client or ExaClient()
        self.max_characters = max_characters
        self.livecrawl = livecrawl

    async def get_contents(
        self,
        urls: List[str],
        max_characters: Optional[int] = None,
    ) -> WebContentsOutput:
        if not urls:
            return WebContentsOutput(pages=[], error="No URLs provided")

        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]

        try:
            response = await self.exa_client.get_contents(
                urls=urls,
                max_characters=max_characters if max_characters is not None else self.max_characters,
                livecrawl=self.livecrawl,
            )

            if response.error:
                return WebContentsOutput(pages=[], error=response.error, request_id=response.request_id)

            pages = []
            for r in response.results:
                pages.append(
                    PageContent(
                        url=r.url,
                        title=r.title,
                        text=r.text,
                        published_date=r.published_date,
                        error=r.error if r.status == "error" else None,
                    )
                )

            return WebContentsOutput(
                pages=pages,
                request_id=response.request_id,
                cost_dollars=response.cost_dollars,
            )

        except Exception as e:
            return WebContentsOutput(pages=[], error=str(e))

    async def execute(self, urls: List[str], max_characters: Optional[int] = None) -> str:
        output = await self.get_contents(urls, max_characters=max_characters)
        return output.to_observation()

