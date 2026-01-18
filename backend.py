"""Backend executor for codex exec --json CLI."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from config import (
    CODEX_EXE_PATH,
    DEFAULT_SANDBOX,
    SANDBOX_MODES,
    MAX_RETRIES,
    RETRY_DELAY,
)
from errors import (
    CodexExecutionError,
    InvalidSandboxModeError,
)
from jsonl_parser import JsonlParser, ParsedResult
from session_manager import Session, get_session_manager

logger = logging.getLogger(__name__)

# Error patterns that indicate a retryable disconnection
RETRYABLE_ERROR_PATTERNS = [
    "Reconnecting",
    "stream disconnected",
    "stream closed",
    "connection reset",
    "connection refused",
    "network error",
]


def _is_retryable_error(errors: list[str]) -> bool:
    """Check if the errors indicate a retryable disconnection."""
    for error in errors:
        error_lower = error.lower()
        for pattern in RETRYABLE_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                return True
    return False


class CodexExecRunner:
    """Executes codex exec --json and parses the output."""

    def __init__(
        self,
        codex_path: Path | None = None,
    ):
        self.codex_path = codex_path or CODEX_EXE_PATH
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
        thread_id: str | None = None,
    ) -> ParsedResult:
        """Execute codex with automatic retry on disconnection."""
        if sandbox not in SANDBOX_MODES:
            raise InvalidSandboxModeError(sandbox, SANDBOX_MODES)

        cwd = cwd or str(Path.cwd())

        last_error = None
        last_result = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{MAX_RETRIES} after {RETRY_DELAY}s delay...")
                await asyncio.sleep(RETRY_DELAY)

            try:
                result = await self._execute_once(prompt, cwd, sandbox, model, thread_id)

                # Check if result has retryable errors but no completion
                if result.errors and _is_retryable_error(result.errors) and not result.completed:
                    logger.warning(f"Retryable error detected: {result.errors}")
                    last_result = result
                    last_error = CodexExecutionError(f"Retryable error: {result.errors}")
                    continue

                return result

            except CodexExecutionError as e:
                logger.warning(f"Execution error on attempt {attempt + 1}: {e}")
                last_error = e
                # Check if it's a retryable error
                if "retryable" in str(e).lower() or attempt < MAX_RETRIES:
                    continue
                raise

        # All retries exhausted
        if last_result and (last_result.agent_messages or last_result.thread_id):
            logger.warning("Returning partial result after retries exhausted")
            return last_result

        if last_error:
            raise last_error

        raise CodexExecutionError("All retry attempts failed")

    async def _execute_once(
        self,
        prompt: str,
        cwd: str,
        sandbox: str,
        model: str | None,
        thread_id: str | None = None,
    ) -> ParsedResult:
        """Execute codex once (single attempt)."""

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

            # Read output until completion
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

            await wait_for_completion()

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
) -> dict[str, Any]:
    """Execute a new codex session."""
    runner = CodexExecRunner()
    result = await runner.execute(
        prompt=prompt,
        cwd=cwd,
        sandbox=sandbox,
        model=model,
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
