import datetime
import sqlite3
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
from typing import List, Optional, Generator

from app.prompts import SYSTEM_MESSAGE

load_dotenv()

MODEL_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DB_CONNECTION_STRING = "sqlite:///conversations.db"


class ChatBot:
    def __init__(self, session_id: Optional[str] = None):
        """Initialize the Chatbot."""
        self.model = init_chat_model(
            model=DEFAULT_MODEL,
            model_provider=MODEL_PROVIDER,
            temperature=0
        )
        self.system_msg = SystemMessage(content=SYSTEM_MESSAGE)

        self._initialize_session(session_id)
        self._initialize_history()

    def _initialize_session(self, session_id: Optional[str] = None) -> None:
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Chat - {timestamp}"

    def _initialize_history(self) -> None:
        """Initialize or load the conversation for the current session."""
        self.history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection=DB_CONNECTION_STRING
        )

        if not self.history.get_messages():
            self.history.add_message(self.system_msg)

    def get_response(self, user_message: str) -> str:
        """Process user message and get AI response."""
        if not (user_message := user_message.strip()):
            return ""

        # Add a user message to the history
        self.history.add_message(HumanMessage(content=user_message))

        # Get response
        response = self.model.invoke(self.history.get_messages())

        # Add AI response to history
        self.history.add_message(AIMessage(content=response.text))

        return response.text

    def stream_response(self, user_message: str) -> Generator[str, None, None]:
        """Stream AI response chunks for a user message."""
        # Check if message is empty
        if not (user_message := user_message.strip()):
            return

        # Add user message to history
        self.history.add_message(HumanMessage(content=user_message))

        # Stream response
        response_content = ""

        for chunk in self.model.stream(
            self.history.get_messages(),
            config={"metadata": {"session_id": self.session_id}},
        ):
            if chunk.text:
                response_content += chunk.text
                yield chunk.text

        # Add final response to history
        self.history.add_message(
            AIMessage(content=response_content)
        )

    def new_session(self) -> None:
        """Create a new conversation."""
        self._initialize_session()
        self._initialize_history()

    def load_session(self, session_id: str) -> None:
        """Load an existing conversation."""
        if not session_id:
            return

        self._initialize_session(session_id)
        self._initialize_history()

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages from the current session."""
        return self.history.get_messages()

    @staticmethod
    def get_previous_conversations() -> List[str]:
        """Get a list of all available conversations."""
        # SQL Query to get previous conversations with more than just the system message
        query = """
        SELECT session_id FROM message_store
        GROUP BY session_id
        HAVING COUNT(*) > 1
        ORDER BY session_id DESC
        """

        # Get the database path from the connection string
        db_path = DB_CONNECTION_STRING.replace("sqlite:///", "")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]


if __name__ == "__main__":
    chatbot = ChatBot()

    user_message = "Patient P102 has a pending cardiology referral."

    for chunk in chatbot.stream_response(user_message):
        print(chunk, end="", flush=True)