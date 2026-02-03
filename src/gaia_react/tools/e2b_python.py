"""
E2B Python Sandbox Tool for code execution.
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution."""

    output: str
    error: Optional[str] = None
    success: bool = True
    execution_time: float = 0.0


class E2BPythonTool:
    """
    E2B Python sandbox for code execution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sandbox_timeout: int = 300,  # 5 minutes sandbox lifetime
        execution_timeout: int = 120,  # 2 minutes per execution
    ):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY not set")

        self.sandbox_timeout = sandbox_timeout
        self.execution_timeout = execution_timeout
        self._sandbox = None

    async def _get_sandbox(self):
        """Get or create the sandbox instance."""
        if self._sandbox is None:
            try:
                from e2b_code_interpreter import AsyncSandbox

                self._sandbox = await AsyncSandbox.create(
                    api_key=self.api_key,
                    timeout=self.sandbox_timeout,
                )
            except ImportError:
                raise ImportError(
                    "e2b-code-interpreter package required. Install with: pip install e2b-code-interpreter"
                )
        return self._sandbox

    async def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in the sandbox.
        """
        import time

        start_time = time.time()

        try:
            async def run_once():
                sb = await self._get_sandbox()
                return await asyncio.wait_for(
                    sb.run_code(code),
                    timeout=self.execution_timeout,
                )

            # Run the code with timeout.
            # Retry once on a common transient E2B failure: sandbox not found (502).
            try:
                try:
                    execution = await run_once()
                except Exception as e:
                    msg = str(e).lower()
                    if "sandbox was not found" in msg or "sandbox not found" in msg or "code\":502" in msg:
                        # Reset sandbox and retry once.
                        self._sandbox = None
                        execution = await run_once()
                    else:
                        raise
            except asyncio.TimeoutError:
                return ExecutionResult(
                    output="",
                    error=f"Execution timed out after {self.execution_timeout}s",
                    success=False,
                    execution_time=time.time() - start_time,
                )

            # Collect output
            output_parts = []

            # Get text result
            if execution.text:
                output_parts.append(execution.text)

            # Get stdout/stderr from logs
            if execution.logs:
                if execution.logs.stdout:
                    for line in execution.logs.stdout:
                        output_parts.append(line)
                if execution.logs.stderr:
                    for line in execution.logs.stderr:
                        output_parts.append(f"[stderr] {line}")

            # Get results (e.g., from display())
            if execution.results:
                for result in execution.results:
                    if hasattr(result, "text") and result.text:
                        output_parts.append(result.text)
                    elif hasattr(result, "data"):
                        output_parts.append(str(result.data))

            output = "\n".join(output_parts) if output_parts else "(no output)"

            # Check for errors
            error = None
            if execution.error:
                error = str(execution.error)
                if hasattr(execution.error, "traceback"):
                    error = execution.error.traceback

            return ExecutionResult(
                output=output,
                error=error,
                success=error is None,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Sandbox error: {str(e)}",
                success=False,
                execution_time=time.time() - start_time,
            )

    async def close(self):
        """Close the sandbox."""
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None

    def execute_sync(self, code: str) -> ExecutionResult:
        """Synchronous wrapper for execute."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.execute(code))
                    return future.result()
            else:
                return loop.run_until_complete(self.execute(code))
        except RuntimeError:
            return asyncio.run(self.execute(code))


def format_execution_result(result: ExecutionResult) -> str:
    """Format execution result as a string for the agent."""
    lines = []

    if result.success:
        lines.append("Code executed successfully.")
    else:
        lines.append("Code execution failed.")

    lines.append(f"Execution time: {result.execution_time:.2f}s")
    lines.append("")

    if result.output:
        lines.append("Output:")
        lines.append(result.output)

    if result.error:
        lines.append("")
        lines.append("Error:")
        lines.append(result.error)

    return "\n".join(lines)

