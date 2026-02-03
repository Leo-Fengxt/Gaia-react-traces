"""Web search tool (Exa)."""

from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass, field

from ..clients.exa import ExaClient, ExaSubpage


@dataclass
class WebSearchResultItem:
    """A single web search result with optional highlights/summary/subpages."""

    title: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    score: float = 0.0
    highlights: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    subpages: List[ExaSubpage] = field(default_factory=list)
    text: str = ""


@dataclass
class WebSearchOutput:
    """Output from web search tool."""

    results: List[WebSearchResultItem]
    error: Optional[str] = None
    request_id: Optional[str] = None
    cost_dollars: Optional[float] = None

    def to_observation(self) -> str:
        """Format as observation string for the agent."""
        if self.error:
            return f"Web search error: {self.error}"

        if not self.results:
            return "No search results found."

        header = []
        if self.request_id:
            header.append(f"EXA requestId: {self.request_id}")
        if self.cost_dollars is not None:
            header.append(f"EXA costDollars: {self.cost_dollars}")
        header_text = (" | ".join(header) + "\n\n") if header else ""

        lines = [header_text + f"Found {len(self.results)} results:\n"]

        for i, result in enumerate(self.results, 1):
            lines.append(f"[{i}] {result.title}")
            lines.append(f"    URL: {result.url}")
            if result.published_date:
                lines.append(f"    Published: {result.published_date}")
            if result.author:
                lines.append(f"    Author: {result.author}")

            if result.summary:
                lines.append(f"    SUMMARY: {result.summary}")

            if result.highlights:
                lines.append("    HIGHLIGHTS:")
                for hl in result.highlights[:5]:
                    hl_text = hl[:500] + "..." if len(hl) > 500 else hl
                    lines.append(f"      - {hl_text}")

            if result.subpages:
                lines.append("    SUBPAGES:")
                for sp in result.subpages[:3]:
                    lines.append(f"      • {sp.title or sp.url}")
                    lines.append(f"        URL: {sp.url}")
                    if sp.summary:
                        sp_summary = sp.summary[:200] + "..." if len(sp.summary) > 200 else sp.summary
                        lines.append(f"        Summary: {sp_summary}")

            if result.text:
                text_preview = result.text[:1000] + "..." if len(result.text) > 1000 else result.text
                lines.append(f"    TEXT_PREVIEW: {text_preview}")

            lines.append("")

        lines.append("─" * 60)
        lines.append("TIP: To read full page content, call web_contents(urls=[...]) with 1-2 specific URLs.")
        lines.append("     Avoid repeating web_search with minor rewording.")
        return "\n".join(lines)


class WebSearchTool:
    """
    Web search tool using Exa API.
    """

    def __init__(
        self,
        exa_client: Optional[ExaClient] = None,
        default_num_results: int = 25,
        default_highlights: bool = True,
        default_summary: bool = True,
        default_subpages: int = 3,
    ):
        self.exa_client = exa_client or ExaClient()
        self.default_num_results = default_num_results
        self.default_highlights = default_highlights
        self.default_summary = default_summary
        self.default_subpages = default_subpages

    async def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: str = "auto",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        category: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        subpages: Optional[int] = None,
        full_text: bool = False,
        max_characters: int = 50000,
        context: Optional[bool] = None,
        context_max_characters: Optional[int] = None,
        include_contents: Optional[bool] = None,
    ) -> WebSearchOutput:
        if not query:
            return WebSearchOutput(results=[], error="No query provided")

        num_results = num_results or self.default_num_results

        if context is not None or context_max_characters is not None or include_contents is not None:
            if context or include_contents:
                full_text = True
            if context_max_characters:
                max_characters = context_max_characters

        use_highlights = highlights if highlights is not None else self.default_highlights
        use_summary = summary if summary is not None else self.default_summary
        use_subpages = subpages if subpages is not None else self.default_subpages

        if isinstance(include_text, str):
            include_text = [include_text]
        if isinstance(exclude_text, str):
            exclude_text = [exclude_text]
        if isinstance(include_text, list):
            include_text = [s for s in include_text if isinstance(s, str) and s.strip()]
            if not include_text:
                include_text = None
        else:
            include_text = None
        if isinstance(exclude_text, list):
            exclude_text = [s for s in exclude_text if isinstance(s, str) and s.strip()]
            if not exclude_text:
                exclude_text = None
        else:
            exclude_text = None

        try:
            response = await self.exa_client.search(
                query=query,
                num_results=num_results,
                search_type=search_type,
                include_text=full_text,
                max_characters=max_characters,
                highlights=use_highlights,
                summary=use_summary,
                subpages=use_subpages,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                category=category,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                include_text_filter=include_text,
                exclude_text_filter=exclude_text,
            )

            if response.error:
                return WebSearchOutput(results=[], error=response.error, request_id=response.request_id)

            results = []
            for r in response.results:
                results.append(
                    WebSearchResultItem(
                        title=r.title,
                        url=r.url,
                        published_date=r.published_date,
                        author=r.author,
                        score=r.score,
                        highlights=r.highlights,
                        summary=r.summary,
                        subpages=r.subpages,
                        text=r.text if full_text else "",
                    )
                )

            return WebSearchOutput(
                results=results,
                request_id=response.request_id,
                cost_dollars=response.cost_dollars,
            )

        except Exception as e:
            return WebSearchOutput(results=[], error=str(e))

    async def execute(
        self,
        query: str,
        num_results: Optional[int] = None,
        **kwargs,
    ) -> str:
        output = await self.search(query, num_results, **kwargs)
        return output.to_observation()

