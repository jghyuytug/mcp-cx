"""Backend executor for codex exec --json CLI."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from config import CODEX_EXE_PATH, DEFAULT_SANDBOX, DEFAULT_TIMEOUT, SANDBOX_MODES
from errors import (
    CodexExecutionError,
    CodexTimeoutError,
    InvalidSandboxModeError,
)
from jsonl_parser import JsonlParser, ParsedResult
from session_manager import Session, get_session_manager

logger = logging.getLogger(__name__)


class CodexExecRunner:
    """Executes codex exec --json and parses the output."""

    def __init__(
        self,
        codex_path: Path | None = None,
        default_timeout: int = DEFAULT_TIMEOUT,
    ):
        self.codex_path = codex_path or CODEX_EXE_PATH
        self.default_timeout = default_timeout
        self._validate_codex_path()

    def _validate_codex_path(self) -> None:
        """Validate that the codex executable exists."""
        if not self.codex_path.exists():
            logger.warning(f"Codex executable not found at: {self.codex_path}")

    def _build_command(
        self,
        sandbox: str,
        model: str | None,
        thread_id: str | None = None,
    ) -> list[str]:
        """Build the codex exec command. Prompt is passed via stdin for UTF-8 support."""
        cmd = [str(self.codex_path), "exec"]

        if thread_id:
            # Resume mode: use stdin (-) for prompt
            cmd.extend(["resume", "--json", "--skip-git-repo-check", thread_id, "-"])
        else:
            # New session mode - use stdin (-) for prompt
            cmd.append("-")  # Read prompt from stdin
            cmd.append("--json")
            cmd.append("--skip-git-repo-check")

            if sandbox:
                cmd.extend(["--sandbox", sandbox])

            if model:
                cmd.extend(["--model", model])

        return cmd

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate a process and its children (cross-platform)."""
        if proc.returncode is not None:
            return

        try:
            if sys.platform == "win32":
                # Windows: use taskkill to terminate process tree
                import subprocess
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                # Unix: send SIGTERM to process group
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                await asyncio.sleep(0.5)
                if proc.returncode is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:
            logger.warning(f"Failed to terminate process: {e}")
            try:
                proc.kill()
            except Exception:
                pass

    async def execute(
        self,
        prompt: str,
        cwd: str | None = None,
        sandbox: str = DEFAULT_SANDBOX,
        model: str | None = None,
        timeout: int | None = None,
        thread_id: str | None = None,
    ) -> ParsedResult:
        """Execute codex and return parsed result."""
        if sandbox not in SANDBOX_MODES:
            raise InvalidSandboxModeError(sandbox, SANDBOX_MODES)

        cwd = cwd or str(Path.cwd())
        timeout = timeout or self.default_timeout

        cmd = self._build_command(sandbox, model, thread_id)
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"Working directory: {cwd}")
        logger.info(f"Prompt (first 200 chars): {prompt[:200]}...")

        parser = JsonlParser()
        stdout_buffer = []
        stderr_buffer = []

        # Set UTF-8 environment
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"

        try:
            # Create subprocess with stdin pipe for prompt
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    start_new_session=True,
                )

            # Write prompt to stdin (UTF-8 encoded)
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()

            # Read output with timeout
            try:
                completion_event = asyncio.Event()

                async def read_stream():
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            break
                        decoded = line.decode("utf-8", errors="replace")
                        stdout_buffer.append(decoded)
                        parser.parse_line(decoded)
                        logger.debug(f"JSONL: {decoded.strip()}")
                        # Stop reading once turn is completed
                        if parser.result.completed:
                            logger.info("Turn completed, stopping stream read")
                            completion_event.set()
                            break

                async def read_stderr():
                    while not completion_event.is_set():
                        try:
                            line = await asyncio.wait_for(
                                proc.stderr.readline(),
                                timeout=0.5
                            )
                            if not line:
                                break
                            stderr_buffer.append(line.decode("utf-8", errors="replace"))
                        except asyncio.TimeoutError:
                            continue

                async def wait_for_completion():
                    await asyncio.gather(read_stream(), read_stderr())
                    # Terminate process after completion (it may not exit on its own)
                    if parser.result.completed:
                        logger.info("Terminating codex process after completion")
                        await self._terminate_process(proc)

                await asyncio.wait_for(wait_for_completion(), timeout=timeout)

            except asyncio.TimeoutError:
                await self._terminate_process(proc)
                partial_output = "".join(stdout_buffer)
                raise CodexTimeoutError(timeout, partial_output)

            # Check exit code
            if proc.returncode != 0:
                stderr = "".join(stderr_buffer)
                logger.error(f"Codex exited with code {proc.returncode}: {stderr}")
                # Still return result if we got any output
                if parser.result.agent_messages or parser.result.thread_id:
                    return parser.get_result()
                raise CodexExecutionError(
                    f"Codex exited with code {proc.returncode}",
                    exit_code=proc.returncode,
                    stderr=stderr,
                )

            return parser.get_result()

        except (asyncio.TimeoutError, CodexTimeoutError):
            raise
        except CodexExecutionError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error executing codex: {e}")
            raise CodexExecutionError(str(e))


async def run_codex(
    prompt: str,
    cwd: str | None = None,
    sandbox: str = DEFAULT_SANDBOX,
    model: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute a new codex session."""
    runner = CodexExecRunner()
    result = await runner.execute(
        prompt=prompt,
        cwd=cwd,
        sandbox=sandbox,
        model=model,
        timeout=timeout,
    )

    # Create session if we got a thread_id
    if result.thread_id:
        session_manager = get_session_manager()
        session = session_manager.create_session(
            thread_id=result.thread_id,
            cwd=cwd or str(Path.cwd()),
            sandbox=sandbox,
            model=model,
        )
        session.add_turn("user", prompt)
        session.add_turn("assistant", result.get_agent_response())
        session_manager.update_session(session)

    return {
        "success": True,
        "threadId": result.thread_id,
        "agent_messages": result.get_agent_response(),
        "reasoning": result.reasoning,
        "completed": result.completed,
        "errors": result.errors if result.errors else None,
    }


async def run_codex_reply(
    thread_id: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Continue an existing codex session."""
    session_manager = get_session_manager()

    # Get session info if available
    session = None
    cwd = None
    sandbox = DEFAULT_SANDBOX
    model = None

    if session_manager.has_session(thread_id):
        session = session_manager.get_session(thread_id)
        cwd = session.cwd
        sandbox = session.sandbox
        model = session.model

    runner = CodexExecRunner()
    result = await runner.execute(
        prompt=prompt,
        cwd=cwd,
        sandbox=sandbox,
        model=model,
        timeout=timeout,
        thread_id=thread_id,
    )

    # Update session
    if session:
        session.add_turn("user", prompt)
        session.add_turn("assistant", result.get_agent_response())
        session_manager.update_session(session)

    return {
        "success": True,
        "threadId": result.thread_id or thread_id,
        "agent_messages": result.get_agent_response(),
        "reasoning": result.reasoning,
        "completed": result.completed,
        "errors": result.errors if result.errors else None,
    }
