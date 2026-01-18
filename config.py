"""Configuration constants for Codex MCP Server."""

import os
from pathlib import Path

# Codex CLI executable path
CODEX_EXE_PATH = Path(
    os.environ.get(
        "CODEX_EXE_PATH",
        r"C:\Users\waw\AppData\Roaming\npm\node_modules\@openai\codex\vendor\x86_64-pc-windows-msvc\codex\codex.exe"
    )
)

# Default timeout for codex exec (in seconds)
DEFAULT_TIMEOUT = 600

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Sandbox modes
SANDBOX_MODES = frozenset({"read-only", "workspace-write", "danger-full-access"})
DEFAULT_SANDBOX = "read-only"

# Default working directory
DEFAULT_CWD = Path.cwd()

# Session storage directory
SESSION_STORAGE_DIR = Path.home() / ".codex-mcp-sessions"

# Logging configuration
LOG_LEVEL = os.environ.get("CODEX_MCP_LOG_LEVEL", "INFO")
