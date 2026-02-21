import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st

class HistoryManager:
    def __init__(self, data_dir: str = "data"):
        """Initialize HistoryManager with data directory"""
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / "chat_history.json"
        self._ensure_data_dir()
        self.history = self._load_history()

    def _ensure_data_dir(self):
        """Ensure data directory and history file exist"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.history_file.exists():
            self._save_to_file({})

    def _load_history(self) -> Dict:
        """Load history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return {}

    def _save_to_file(self, data: Dict):
        """Save data to JSON file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")

    def create_new_session(self) -> str:
        """Create a new chat session and return its ID"""
        session_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history[session_id] = {
            "created_at": timestamp,
            "updated_at": timestamp,
            "title": f"New Chat - {timestamp}",
            "messages": []
        }
        self._save_to_file(self.history)
        return session_id

    def save_session(self, session_id: str, messages: List[Dict]):
        """Save messages to a specific session"""
        if session_id not in self.history:
            self.create_new_session() # Should ideally not happen if flow is correct, but safety net
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a title from the first user message if it's the default title
        current_title = self.history[session_id].get("title", "")
        if "New Chat" in current_title and messages:
            first_user_msg = next((m for m in messages if m["role"] == "user"), None)
            if first_user_msg:
                # Truncate to 30 chars
                new_title = first_user_msg["content"][:30] + "..." if len(first_user_msg["content"]) > 30 else first_user_msg["content"]
                self.history[session_id]["title"] = new_title

        self.history[session_id]["updated_at"] = timestamp
        self.history[session_id]["messages"] = messages
        self._save_to_file(self.history)

    def get_session(self, session_id: str) -> Dict:
        """Get a specific session"""
        return self.history.get(session_id, {})

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions sorted by updated_at (newest first)"""
        sessions = []
        for sess_id, data in self.history.items():
            sessions.append({
                "id": sess_id,
                "title": data.get("title", "Untitled Chat"),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", []))
            })
        
        # Sort by updated_at descending
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.history:
            del self.history[session_id]
            self._save_to_file(self.history)
