import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    duration_ms: float = 0.0


@dataclass
class Interaction:
    user_input: str = ""
    llm_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    end_time: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "llm_response": self.llm_response,
            "tool_calls": [tc.__dict__ for tc in self.tool_calls],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }


class Session:
    def __init__(
        self, session_id: Optional[str] = None, max_history: int = MAX_HISTORY_MESSAGES
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        self.messages: list[Message] = []
        self.interactions: list[Interaction] = []
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.current_interaction: Optional[Interaction] = None
        self._closed = False

    def add_message(self, role: str, content: str) -> None:
        if self._closed:
            return
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]
        self.last_activity = datetime.utcnow()
        logger.debug(f"Session {self.session_id}: Added {role} message")

    def get_messages(self) -> list[dict[str, str]]:
        return [msg.to_dict() for msg in self.messages]

    def start_interaction(self) -> Interaction:
        if self._closed:
            return Interaction()
        self.current_interaction = Interaction()
        self.last_activity = datetime.utcnow()
        logger.debug(f"Session {self.session_id}: Started interaction")
        return self.current_interaction

    def end_interaction(self, user_input: str, llm_response: str) -> None:
        if self._closed or not self.current_interaction:
            return
        self.current_interaction.user_input = user_input
        self.current_interaction.llm_response = llm_response
        self.current_interaction.end_time = datetime.utcnow().isoformat() + "Z"
        start_dt = datetime.fromisoformat(
            self.current_interaction.start_time.replace("Z", "+00:00")
        )
        self.current_interaction.duration_ms = (
            datetime.utcnow() - start_dt.replace(tzinfo=None)
        ).total_seconds() * 1000

        self.interactions.append(self.current_interaction)
        self.current_interaction = None
        self.last_activity = datetime.utcnow()
        logger.debug(f"Session {self.session_id}: Ended interaction")

    def add_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: float = 0.0,
    ) -> None:
        if self._closed:
            return
        tool_call = ToolCallRecord(
            name=name,
            arguments=arguments,
            result=result,
            success=success,
            duration_ms=duration_ms,
        )
        if self.current_interaction:
            self.current_interaction.tool_calls.append(tool_call)
        self.last_activity = datetime.utcnow()
        logger.debug(f"Session {self.session_id}: Added tool call {name}")

    def get_stats(self) -> dict[str, Any]:
        now = datetime.utcnow()
        age_seconds = (now - self.created_at).total_seconds()
        idle_seconds = (now - self.last_activity).total_seconds()

        total_duration_ms = sum(i.duration_ms for i in self.interactions)
        avg_duration_ms = (
            total_duration_ms / len(self.interactions) if self.interactions else 0.0
        )

        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "interaction_count": len(self.interactions),
            "total_duration_ms": total_duration_ms,
            "avg_interaction_duration_ms": avg_duration_ms,
            "age_seconds": age_seconds,
            "idle_seconds": idle_seconds,
            "created_at": self.created_at.isoformat() + "Z",
            "last_activity": self.last_activity.isoformat() + "Z",
        }

    def is_expired(self, timeout_seconds: int) -> bool:
        idle_seconds = (datetime.utcnow() - self.last_activity).total_seconds()
        return idle_seconds > timeout_seconds

    def close(self) -> None:
        self._closed = True
        logger.info(f"Session {self.session_id}: Closed")


class SessionManager:
    def __init__(self, timeout_seconds: int = 3600):
        self.sessions: dict[str, Session] = {}
        self.timeout_seconds = timeout_seconds

    def create_session(self, session_id: Optional[str] = None) -> Session:
        session = Session(session_id=session_id)
        self.sessions[session.session_id] = session
        logger.info(f"Created session {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def restore_session(self, session_id: str) -> Optional[Session]:
        session = self.get_session(session_id)
        if session and not session.is_expired(self.timeout_seconds):
            session.last_activity = datetime.utcnow()
            logger.info(f"Restored session {session_id}")
            return session
        return None

    def end_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            session.close()
            logger.info(f"Ended session {session_id}")

    def cleanup_expired(self) -> int:
        expired = [
            sid
            for sid, session in self.sessions.items()
            if session.is_expired(self.timeout_seconds)
        ]
        for sid in expired:
            self.end_session(sid)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(
                [
                    s
                    for s in self.sessions.values()
                    if not s.is_expired(self.timeout_seconds)
                ]
            ),
        }
