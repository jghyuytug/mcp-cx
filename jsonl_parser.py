"""JSONL event stream parser for codex exec --json output."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from errors import ParseError

logger = logging.getLogger(__name__)


@dataclass
class CodexEvent:
    """Represents a parsed event from codex exec --json."""

    event_type: str
    data: dict[str, Any]
    raw: str


@dataclass
class ParsedResult:
    """Aggregated result from parsing codex exec output."""

    thread_id: str | None = None
    agent_messages: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    command_executions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    completed: bool = False
    raw_events: list[CodexEvent] = field(default_factory=list)

    def get_agent_response(self) -> str:
        """Get the combined agent response text."""
        return "\n\n".join(self.agent_messages) if self.agent_messages else ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "thread_id": self.thread_id,
            "agent_messages": self.get_agent_response(),
            "reasoning": self.reasoning,
            "command_executions": self.command_executions,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "completed": self.completed,
        }


class JsonlParser:
    """Parser for codex exec --json JSONL output stream."""

    def __init__(self):
        self.result = ParsedResult()

    def parse_line(self, line: str) -> CodexEvent | None:
        """Parse a single JSONL line into a CodexEvent."""
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSONL line: {line[:100]}... Error: {e}")
            return None

        event_type = data.get("type", "unknown")
        event = CodexEvent(event_type=event_type, data=data, raw=line)
        self._process_event(event)
        return event

    def _process_event(self, event: CodexEvent) -> None:
        """Process an event and update the result."""
        self.result.raw_events.append(event)
        handler = getattr(self, f"_handle_{event.event_type.replace('.', '_')}", None)
        if handler:
            handler(event.data)
        else:
            logger.debug(f"No handler for event type: {event.event_type}")

    def _handle_thread_started(self, data: dict[str, Any]) -> None:
        """Handle thread.started event."""
        self.result.thread_id = data.get("thread_id") or data.get("threadId")
        logger.info(f"Thread started: {self.result.thread_id}")

    def _handle_item_completed(self, data: dict[str, Any]) -> None:
        """Handle item.completed event."""
        item = data.get("item", {})
        item_type = item.get("type", "")

        if item_type == "message":
            self._extract_message_content(item)
        elif item_type == "agent_message":
            # Direct agent message (simpler format)
            text = item.get("text", "")
            if text:
                self.result.agent_messages.append(text)
        elif item_type == "reasoning":
            text = item.get("text", "")
            if text:
                self.result.reasoning.append(text)
        elif item_type == "function_call":
            self._extract_function_call(item)
        elif item_type == "function_call_output":
            self._extract_function_output(item)

    def _handle_turn_completed(self, data: dict[str, Any]) -> None:
        """Handle turn.completed event."""
        self.result.completed = True
        logger.info("Turn completed")

    def _handle_response_completed(self, data: dict[str, Any]) -> None:
        """Handle response.completed event (alias for turn completion)."""
        self.result.completed = True
        logger.info("Response completed")

    def _handle_error(self, data: dict[str, Any]) -> None:
        """Handle error events."""
        error_msg = data.get("message") or data.get("error") or str(data)
        self.result.errors.append(error_msg)
        logger.error(f"Codex error: {error_msg}")

    def _extract_message_content(self, item: dict[str, Any]) -> None:
        """Extract message content from an item."""
        role = item.get("role", "")
        content = item.get("content", [])

        if role == "assistant":
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "text":
                        text = part.get("text", "")
                        if text:
                            self.result.agent_messages.append(text)
                    elif part_type == "reasoning":
                        reasoning = part.get("text", "") or part.get("content", "")
                        if reasoning:
                            self.result.reasoning.append(reasoning)
                elif isinstance(part, str):
                    self.result.agent_messages.append(part)

    def _extract_function_call(self, item: dict[str, Any]) -> None:
        """Extract function call information."""
        self.result.tool_calls.append({
            "name": item.get("name", ""),
            "arguments": item.get("arguments", {}),
            "call_id": item.get("call_id", ""),
        })

    def _extract_function_output(self, item: dict[str, Any]) -> None:
        """Extract function call output."""
        output = item.get("output", "")
        call_id = item.get("call_id", "")
        if call_id and output:
            self.result.command_executions.append({
                "call_id": call_id,
                "output": output,
            })

    def parse_stream(self, stream: str) -> ParsedResult:
        """Parse complete JSONL stream."""
        for line in stream.splitlines():
            self.parse_line(line)
        return self.result

    def get_result(self) -> ParsedResult:
        """Get the current parsed result."""
        return self.result


def parse_jsonl_output(output: str) -> ParsedResult:
    """Convenience function to parse JSONL output."""
    parser = JsonlParser()
    return parser.parse_stream(output)
