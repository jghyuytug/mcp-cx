"""Custom exceptions for Codex MCP Server."""


class CodexMCPError(Exception):
    """Base exception for Codex MCP Server."""
    pass


class CodexExecutionError(CodexMCPError):
    """Raised when codex exec fails to run."""

    def __init__(self, message: str, exit_code: int | None = None, stderr: str | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class CodexTimeoutError(CodexMCPError):
    """Raised when codex exec times out."""

    def __init__(self, timeout: int, partial_output: str | None = None):
        super().__init__(f"Codex execution timed out after {timeout} seconds")
        self.timeout = timeout
        self.partial_output = partial_output


class SessionNotFoundError(CodexMCPError):
    """Raised when attempting to resume a non-existent session."""

    def __init__(self, thread_id: str):
        super().__init__(f"Session not found: {thread_id}")
        self.thread_id = thread_id


class InvalidSandboxModeError(CodexMCPError):
    """Raised when an invalid sandbox mode is specified."""

    def __init__(self, mode: str, valid_modes: frozenset[str]):
        super().__init__(f"Invalid sandbox mode: {mode}. Valid modes: {', '.join(sorted(valid_modes))}")
        self.mode = mode
        self.valid_modes = valid_modes


class ParseError(CodexMCPError):
    """Raised when JSONL parsing fails."""

    def __init__(self, message: str, raw_line: str | None = None):
        super().__init__(message)
        self.raw_line = raw_line
