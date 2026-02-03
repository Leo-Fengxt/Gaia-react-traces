"""
Disabled tool stubs.
"""

from __future__ import annotations

from typing import Any

from .e2b_python import ExecutionResult


class DisabledWebSearchTool:
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        return "Error: web_search tool is disabled by --allowed-tools."


class DisabledWebContentsTool:
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        return "Error: web_contents tool is disabled by --allowed-tools."


class DisabledE2BPythonTool:
    async def execute(self, code: str) -> ExecutionResult:
        return ExecutionResult(
            output="",
            error="execute_python tool is disabled by --allowed-tools.",
            success=False,
            execution_time=0.0,
        )

    async def close(self) -> None:
        return None

