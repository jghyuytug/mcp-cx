"""Session manager for Codex MCP Server."""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SESSION_STORAGE_DIR
from errors import SessionNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a Codex session."""

    thread_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    cwd: str = ""
    sandbox: str = "read-only"
    model: str | None = None
    turn_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "cwd": self.cwd,
            "sandbox": self.sandbox,
            "model": self.model,
            "turn_count": self.turn_count,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            cwd=data.get("cwd", ""),
            sandbox=data.get("sandbox", "read-only"),
            model=data.get("model"),
            turn_count=data.get("turn_count", 0),
            history=data.get("history", []),
        )

    def add_turn(self, role: str, content: str) -> None:
        """Add a turn to the session history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.turn_count += 1
        self.last_active = datetime.now()


class SessionManager:
    """Manages Codex sessions with persistence."""

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir or SESSION_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._load_sessions()

    def _get_session_file(self, thread_id: str) -> Path:
        """Get the file path for a session."""
        safe_id = thread_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def _load_sessions(self) -> None:
        """Load all sessions from disk."""
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = Session.from_dict(data)
                    self._sessions[session.thread_id] = session
            except Exception as e:
                logger.warning(f"Failed to load session from {file}: {e}")

    def _save_session(self, session: Session) -> None:
        """Save a session to disk."""
        file = self._get_session_file(session.thread_id)
        try:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save session {session.thread_id}: {e}")

    def create_session(
        self,
        thread_id: str,
        cwd: str = "",
        sandbox: str = "read-only",
        model: str | None = None,
    ) -> Session:
        """Create a new session."""
        with self._lock:
            session = Session(
                thread_id=thread_id,
                cwd=cwd,
                sandbox=sandbox,
                model=model,
            )
            self._sessions[thread_id] = session
            self._save_session(session)
            logger.info(f"Created session: {thread_id}")
            return session

    def get_session(self, thread_id: str) -> Session:
        """Get an existing session by thread_id."""
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                raise SessionNotFoundError(thread_id)
            return session

    def has_session(self, thread_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return thread_id in self._sessions

    def update_session(self, session: Session) -> None:
        """Update a session."""
        with self._lock:
            session.last_active = datetime.now()
            self._sessions[session.thread_id] = session
            self._save_session(session)

    def delete_session(self, thread_id: str) -> None:
        """Delete a session."""
        with self._lock:
            if thread_id in self._sessions:
                del self._sessions[thread_id]
                file = self._get_session_file(thread_id)
                if file.exists():
                    file.unlink()
                logger.info(f"Deleted session: {thread_id}")

    def list_sessions(self) -> list[Session]:
        """List all sessions."""
        with self._lock:
            return list(self._sessions.values())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        now = datetime.now()
        removed = 0
        with self._lock:
            to_remove = []
            for thread_id, session in self._sessions.items():
                age = (now - session.last_active).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(thread_id)

            for thread_id in to_remove:
                del self._sessions[thread_id]
                file = self._get_session_file(thread_id)
                if file.exists():
                    file.unlink()
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")
        return removed


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
