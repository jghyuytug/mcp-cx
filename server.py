"""MCP Server for Codex - wraps codex exec --json CLI."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from config import DEFAULT_SANDBOX, DEFAULT_TIMEOUT, SANDBOX_MODES, LOG_LEVEL
from errors import (
    CodexExecutionError,
    CodexTimeoutError,
    CodexMCPError,
    SessionNotFoundError,
)
from backend import run_codex, run_codex_reply

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Path.home() / ".codex-mcp-server.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("codex-mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="codex",
            description=(
                "Create a new Codex session and execute a coding task. "
                "Codex is an AI coding assistant that can analyze code, answer questions, "
                "and provide code suggestions. Use this tool for starting new conversations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question for Codex to process",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the session. Defaults to current directory.",
                    },
                    "sandbox": {
                        "type": "string",
                        "enum": list(SANDBOX_MODES),
                        "default": DEFAULT_SANDBOX,
                        "description": "Sandbox mode: 'read-only' (default, safest), 'workspace-write', or 'danger-full-access'",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name override (e.g., 'gpt-4o', 'o3-mini')",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": DEFAULT_TIMEOUT,
                        "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="codex-reply",
            description=(
                "Continue an existing Codex conversation. "
                "Use this to send follow-up messages in an ongoing session. "
                "Requires the threadId from a previous codex call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "threadId": {
                        "type": "string",
                        "description": "The session/thread ID from a previous codex call",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The follow-up message or question",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": DEFAULT_TIMEOUT,
                        "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
                    },
                },
                "required": ["threadId", "prompt"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool call: {name} with arguments: {arguments}")

    try:
        if name == "codex":
            result = await run_codex(
                prompt=arguments["prompt"],
                cwd=arguments.get("cwd"),
                sandbox=arguments.get("sandbox", DEFAULT_SANDBOX),
                model=arguments.get("model"),
                timeout=arguments.get("timeout", DEFAULT_TIMEOUT),
            )
        elif name == "codex-reply":
            result = await run_codex_reply(
                thread_id=arguments["threadId"],
                prompt=arguments["prompt"],
                timeout=arguments.get("timeout", DEFAULT_TIMEOUT),
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Format the response
        response_parts = []

        if result.get("agent_messages"):
            response_parts.append(result["agent_messages"])

        if result.get("errors"):
            response_parts.append(f"\n\n**Errors:**\n{chr(10).join(result['errors'])}")

        if result.get("threadId"):
            response_parts.append(f"\n\n---\n*Thread ID: {result['threadId']}*")

        response_text = "".join(response_parts) if response_parts else "No response from Codex."
        logger.info(f"Tool response length: {len(response_text)}")

        return [TextContent(type="text", text=response_text)]

    except CodexTimeoutError as e:
        logger.error(f"Timeout error: {e}")
        error_msg = f"**Timeout Error:** Codex execution timed out after {e.timeout} seconds."
        if e.partial_output:
            error_msg += f"\n\n**Partial Output:**\n{e.partial_output[:1000]}..."
        return [TextContent(type="text", text=error_msg)]

    except SessionNotFoundError as e:
        logger.error(f"Session not found: {e}")
        return [TextContent(
            type="text",
            text=f"**Session Not Found:** Thread ID '{e.thread_id}' not found. Please start a new session with 'codex' tool.",
        )]

    except CodexExecutionError as e:
        logger.error(f"Execution error: {e}")
        error_msg = f"**Execution Error:** {str(e)}"
        if e.stderr:
            error_msg += f"\n\n**Stderr:**\n{e.stderr}"
        return [TextContent(type="text", text=error_msg)]

    except CodexMCPError as e:
        logger.error(f"MCP error: {e}")
        return [TextContent(type="text", text=f"**Error:** {str(e)}")]

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [TextContent(type="text", text=f"**Unexpected Error:** {str(e)}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting Codex MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
