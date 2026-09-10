from typing import Dict, List


class ConversationMemory:

    def __init__(self):

        self.memory: Dict[str, List[dict]] = {}

    def create_session(self, session_id: str):

        if session_id not in self.memory:
            self.memory[session_id] = []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):

        self.create_session(session_id)

        self.memory[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str):

        return self.memory.get(session_id, [])

    def clear_session(self, session_id: str):

        self.memory.pop(session_id, None)


conversation_memory = ConversationMemory()