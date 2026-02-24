"""Code Tool — Wrapper around the existing Synthesis + Sandbox pipeline.

This tool accepts a task description, generates code via the Synthesis
node, and executes it via the Sandbox node.  It is a convenience
wrapper for the LLM-driven code generation → execution loop.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


class CodeTool(IsaacTool):
    """Generate and execute code via the Synthesis → Sandbox pipeline."""

    name = "code"
    description = (
        "Generate Python code for a given task and execute it in the sandbox. "
        "Wraps Synthesis + Sandbox nodes."
    )
    risk_level = 3
    requires_approval = False
    sandbox_required = True

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate code and run it in the sandbox.

        Parameters
        ----------
        task:
            A natural-language description of what code to write.
        code:
            Direct Python code to execute (skips generation).
        timeout:
            Execution timeout in seconds (default 30).
        """
        code: str = kwargs.get("code", "")
        task: str = kwargs.get("task", "")
        timeout: int = int(kwargs.get("timeout", 30))

        # If code is provided directly, skip generation
        if not code and not task:
            return ToolResult(
                success=False,
                error="Provide either 'code' (direct Python) or 'task' (description for LLM).",
            )

        if not code:
            gen_result = await self._generate(task)
            if not gen_result.success:
                return gen_result
            code = gen_result.output or ""

        return await self._execute_code(code, timeout)

    async def _generate(self, task: str) -> ToolResult:
        """Use the LLM to generate Python code for the task."""
        try:
            from isaac.llm.router import get_router, TaskComplexity

            router = get_router()
            llm = router.route(TaskComplexity.MODERATE)

            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(
                    content=(
                        "You are a Python code generator. Given a task, output ONLY "
                        "valid Python code. No markdown fences, no explanations — "
                        "just the code that should be executed."
                    )
                ),
                HumanMessage(content=f"Task: {task}"),
            ]
            response = llm.invoke(messages)
            code_text = response.content
            if isinstance(code_text, list):
                code_text = "\n".join(str(c) for c in code_text)
            code_text = str(code_text).strip()

            # Strip markdown fences if LLM included them
            if code_text.startswith("```"):
                lines = code_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                code_text = "\n".join(lines)

            return ToolResult(success=True, output=code_text)
        except Exception as exc:
            logger.error("Code generation failed: %s", exc)
            return ToolResult(success=False, error=f"Code generation failed: {exc}")

    async def _execute_code(self, code: str, timeout: int) -> ToolResult:
        """Run generated code in the sandbox."""
        try:
            from isaac.sandbox.executor import CodeExecutor

            executor = CodeExecutor()
            try:
                result = executor.execute(code)
            finally:
                executor.close()

            return ToolResult(
                success=result.exit_code == 0,
                output=result.stdout,
                error=result.stderr if result.exit_code != 0 else None,
            )
        except Exception as exc:
            logger.error("Code execution failed: %s", exc)
            return ToolResult(success=False, error=str(exc))
