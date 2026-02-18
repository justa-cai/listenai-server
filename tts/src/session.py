"""Session management for TTS requests."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any


@dataclass
class Session:
    """TTS request session."""

    request_id: str
    connection_id: str
    params: dict
    created_at: datetime = field(default_factory=datetime.now)
    task: Optional[Any] = None  # TTSRequestTask reference
    result: Optional[dict] = None
    cancelled: bool = False
    state: str = "created"  # created, queued, processing, completed, failed, cancelled

    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            "request_id": self.request_id,
            "connection_id": self.connection_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state,
            "cancelled": self.cancelled,
            "has_result": self.result is not None,
        }


class SessionManager:
    """Manager for TTS sessions."""

    def __init__(self):
        """Initialize the session manager."""
        self._sessions: dict[str, Session] = {}

    def create_session(
        self,
        request_id: str,
        connection_id: str,
        params: dict
    ) -> Session:
        """
        Create a new session.

        Args:
            request_id: Unique request identifier
            connection_id: Connection identifier
            params: Request parameters

        Returns:
            Created Session object
        """
        session = Session(
            request_id=request_id,
            connection_id=connection_id,
            params=params
        )
        self._sessions[request_id] = session
        return session

    def get_session(self, request_id: str) -> Optional[Session]:
        """
        Get an existing session.

        Args:
            request_id: Request identifier

        Returns:
            Session object or None if not found
        """
        return self._sessions.get(request_id)

    def remove_session(self, request_id: str) -> Optional[Session]:
        """
        Remove a session.

        Args:
            request_id: Request identifier

        Returns:
            Removed Session object or None if not found
        """
        return self._sessions.pop(request_id, None)

    def update_session_state(
        self,
        request_id: str,
        state: str
    ) -> Optional[Session]:
        """
        Update session state.

        Args:
            request_id: Request identifier
            state: New state

        Returns:
            Updated Session object or None if not found
        """
        session = self.get_session(request_id)
        if session:
            session.state = state
        return session

    def get_sessions_by_connection(
        self,
        connection_id: str
    ) -> list[Session]:
        """
        Get all sessions for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            List of Session objects
        """
        return [
            s for s in self._sessions.values()
            if s.connection_id == connection_id
        ]

    def cleanup_connection_sessions(self, connection_id: str) -> list[Session]:
        """
        Remove all sessions for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            List of removed Session objects
        """
        removed = []
        to_remove = [
            request_id for request_id, session in self._sessions.items()
            if session.connection_id == connection_id
        ]
        for request_id in to_remove:
            session = self.remove_session(request_id)
            if session:
                removed.append(session)
        return removed

    @property
    def session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)

    def get_all_sessions(self) -> list[Session]:
        """Get all active sessions."""
        return list(self._sessions.values())
