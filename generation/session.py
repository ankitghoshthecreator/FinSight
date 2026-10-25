"""Multi-turn conversational session memory for financial analysts.
Allows contextual follow-up questions (e.g. 'What was the SLR?' followed by 'How does that compare to last year?').
"""

import time
import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str = Field(description="user | assistant")
    content: str
    timestamp: float = Field(default_factory=time.time)


class Session(BaseModel):
    session_id: str
    turns: List[Turn] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class SessionManager:
    """Thread-safe conversation session registry with sliding-window history."""

    def __init__(self, max_history_turns: int = 6):
        self._sessions: Dict[str, Session] = {}
        self.max_history_turns = max_history_turns

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if not session_id or session_id not in self._sessions:
            new_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            self._sessions[new_id] = Session(session_id=new_id)
            return self._sessions[new_id]
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> Session:
        session = self.get_or_create(session_id)
        session.turns.append(Turn(role=role, content=content))
        session.updated_at = time.time()
        # Retain last N turns to fit within context limits
        if len(session.turns) > self.max_history_turns:
            session.turns = session.turns[-self.max_history_turns:]
        return session

    def get_formatted_history(self, session_id: str) -> str:
        """Formats conversation history for LLM prompt context injection."""
        if session_id not in self._sessions:
            return ""
        session = self._sessions[session_id]
        if not session.turns:
            return ""
        formatted = []
        for turn in session.turns:
            speaker = "Analyst" if turn.role == "user" else "FinSight"
            formatted.append(f"{speaker}: {turn.content}")
        return "\n".join(formatted)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def count(self) -> int:
        return len(self._sessions)


session_manager = SessionManager()
