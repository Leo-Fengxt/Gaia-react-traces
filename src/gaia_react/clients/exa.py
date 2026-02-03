"""
Exa AI API Client for search and content retrieval.

This client uses the /search and /contents endpoints.

The search endpoint can optionally include:
- highlights: LLM-selected relevant snippets (low-cost, high-signal)
- summary: LLM-generated page summary (low-cost, high-signal)
- subpages: Related subpages with their own highlights/summaries

These are lightweight features that massively reduce context size
compared to fetching full page text, while preserving key information.
Use web_contents for full text when highlights/summary are insufficient.
"""

from __future__ import annotations

import os
import json
import aiohttp
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ExaSubpage:
    """A subpage linked from a search result."""

    url: str
    title: str = ""
    summary: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    author: Optional[str] = None


@dataclass
class ExaSearchResult:
    """A single search result from EXA."""

    title: str
    url: str
    score: float = 0.0
    published_date: Optional[str] = None
    author: Optional[str] = None
    id: Optional[str] = None
    text: str = ""  # Full page text content (if requested)
    # Lightweight LLM-extracted fields (low-cost, high-signal)
    highlights: List[str] = field(default_factory=list)
    highlight_scores: List[float] = field(default_factory=list)
    summary: Optional[str] = None
    subpages: List[ExaSubpage] = field(default_factory=list)


@dataclass
class ExaContentResult:
    """Content retrieved from a URL via EXA."""

    url: str
    title: str
    text: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None


@dataclass
class ExaSearchResponse:
    """Response from EXA /search endpoint."""

    results: List[ExaSearchResult] = field(default_factory=list)
    request_id: Optional[str] = None
    cost_dollars: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ExaContentsResponse:
    """Response from EXA /contents endpoint."""

    results: List[ExaContentResult] = field(default_factory=list)
    request_id: Optional[str] = None
    cost_dollars: Optional[float] = None
    error: Optional[str] = None


class ExaClient:
    """
    EXA API client for search and content retrieval.

    Endpoints used:
    - /search: Semantic or keyword search (with optional highlights/summary/subpages)
    - /contents: Full text retrieval from specific URLs

    Recommended usage pattern:
    1. Use search with highlights=True, summary=True for initial exploration (low-cost, high-signal)
    2. Use get_contents for URLs that need full text inspection
    """

    BASE_URL = "https://api.exa.ai"
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the EXA client.

        Args:
            api_key: EXA API key. If not provided, reads from EXA_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EXA API key required. Set EXA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.timeout = timeout
        self._total_cost = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost of API calls in dollars."""
        return self._total_cost

    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make a POST request to the EXA API.
        """
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}

                    data = await response.json()

                    # Track costs
                    if "costDollars" in data and data["costDollars"]:
                        cost = data["costDollars"]
                        if isinstance(cost, dict) and "total" in cost:
                            self._total_cost += cost["total"]
                        elif isinstance(cost, (int, float)):
                            self._total_cost += cost

                    return data

        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {self.timeout}s"}
        except aiohttp.ClientError as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse response JSON"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    async def search(
        self,
        query: str,
        num_results: int = 25,
        search_type: str = "auto",
        include_text: bool = False,
        max_characters: int = 5000,
        highlights: bool = True,
        highlights_num_sentences: int = 5,
        summary: bool = True,
        subpages: int = 3,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_text_filter: Optional[List[str]] = None,
        exclude_text_filter: Optional[List[str]] = None,
    ) -> ExaSearchResponse:
        """
        Search using EXA's /search endpoint.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
            "useAutoprompt": False,
        }

        # Build contents options - always request something
        contents: Dict[str, Any] = {
            "livecrawl": "fallback",
        }

        # Full text (expensive, only when explicitly requested)
        if include_text:
            contents["text"] = {"maxCharacters": max_characters}

        # Highlights (low-cost, high-signal)
        if highlights:
            contents["highlights"] = {"numSentences": highlights_num_sentences}

        # Summary (low-cost, high-signal)
        if summary:
            contents["summary"] = True

        # Subpages (discover related content) - just a number, not an object
        if subpages and subpages > 0:
            contents["subpages"] = subpages

        if contents:
            payload["contents"] = contents

        # Add optional filters
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if category:
            payload["category"] = category
        if include_text_filter:
            payload["includeText"] = include_text_filter
        if exclude_text_filter:
            payload["excludeText"] = exclude_text_filter

        data = await self._make_request("/search", payload)

        if "error" in data:
            return ExaSearchResponse(error=data["error"])

        # Parse results
        results = []
        for item in data.get("results", []):
            # Parse subpages
            subpages_list = []
            for sp in item.get("subpages", []):
                subpages_list.append(
                    ExaSubpage(
                        url=sp.get("url", ""),
                        title=sp.get("title", ""),
                        summary=sp.get("summary"),
                        highlights=sp.get("highlights", []),
                        published_date=sp.get("publishedDate"),
                        author=sp.get("author"),
                    )
                )

            results.append(
                ExaSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    score=item.get("score", 0.0),
                    published_date=item.get("publishedDate"),
                    author=item.get("author"),
                    id=item.get("id"),
                    text=item.get("text", ""),
                    highlights=item.get("highlights", []),
                    highlight_scores=item.get("highlightScores", []),
                    summary=item.get("summary"),
                    subpages=subpages_list,
                )
            )

        return ExaSearchResponse(
            results=results,
            request_id=data.get("requestId"),
            cost_dollars=data.get("costDollars", {}).get("total")
            if isinstance(data.get("costDollars"), dict)
            else data.get("costDollars"),
        )

    async def get_contents(
        self,
        urls: List[str],
        max_characters: int = 150000,
        livecrawl: str = "fallback",
    ) -> ExaContentsResponse:
        """
        Get full page contents using EXA's /contents endpoint.
        """
        payload: Dict[str, Any] = {
            "urls": urls,
            "text": {
                "maxCharacters": max_characters,
            },
            "livecrawl": livecrawl,
        }

        data = await self._make_request("/contents", payload)

        if "error" in data:
            return ExaContentsResponse(error=data["error"])

        # Parse results
        results = []
        for item in data.get("results", []):
            results.append(
                ExaContentResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    text=item.get("text", ""),
                    published_date=item.get("publishedDate"),
                    author=item.get("author"),
                )
            )

        # Parse statuses for errors
        statuses = data.get("statuses", [])
        for status_info in statuses:
            url = status_info.get("id", "")
            if status_info.get("status") == "error":
                error_info = status_info.get("error", {})
                # Find corresponding result and mark as error
                found = False
                for result in results:
                    if result.url == url:
                        result.status = "error"
                        result.error = error_info.get("tag", "Unknown error")
                        found = True
                        break

                if not found:
                    # Add error result if not found
                    results.append(
                        ExaContentResult(
                            url=url,
                            title="",
                            text="",
                            status="error",
                            error=error_info.get("tag", "Unknown error"),
                        )
                    )

        return ExaContentsResponse(
            results=results,
            request_id=data.get("requestId"),
            cost_dollars=data.get("costDollars", {}).get("total")
            if isinstance(data.get("costDollars"), dict)
            else data.get("costDollars"),
        )


def format_search_results(response: ExaSearchResponse) -> str:
    """Format search results as a string for the agent."""
    if response.error:
        return f"Search error: {response.error}"

    if not response.results:
        return "No search results found."

    lines = [f"Found {len(response.results)} results:\n"]

    for i, result in enumerate(response.results, 1):
        lines.append(f"[{i}] {result.title}")
        lines.append(f"    URL: {result.url}")
        if result.published_date:
            lines.append(f"    Published: {result.published_date}")
        if result.text:
            # Include snippet of text
            snippet = result.text[:500] + "..." if len(result.text) > 500 else result.text
            lines.append(f"    Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines)


def format_contents_results(response: ExaContentsResponse) -> str:
    """Format contents results as a string for the agent."""
    if response.error:
        return f"Contents error: {response.error}"

    if not response.results:
        return "No content retrieved."

    lines = []

    for result in response.results:
        lines.append(f"=== {result.title} ===")
        lines.append(f"URL: {result.url}")

        if result.status == "error":
            lines.append(f"ERROR: {result.error}")
        else:
            lines.append(f"Content ({len(result.text)} chars):")
            lines.append(result.text)

        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


# Import asyncio for the timeout error
import asyncio

