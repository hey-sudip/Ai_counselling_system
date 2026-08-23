"""In-memory / pluggable store for per-session emotion timelines (Phase 10)."""
from collections import defaultdict


class SessionStore:
    def __init__(self):
        self._sessions = defaultdict(list)

    def add_snapshot(self, session_id, timestamp, probabilities):
        self._sessions[session_id].append({
            "timestamp": timestamp,
            "probabilities": probabilities,
        })

    def get_timeline(self, session_id):
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id):
        self._sessions.pop(session_id, None)


session_store = SessionStore()
