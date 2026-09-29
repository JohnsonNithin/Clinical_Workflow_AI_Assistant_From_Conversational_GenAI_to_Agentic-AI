from dotenv import load_dotenv
import os
import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from typing import List, Optional
from app.prompts import SYSTEM_MESSAGE as SM


# Load environment variables from .env
load_dotenv()
#print(bool(os.getenv("clinical_AI")) ) 
MODEL_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DB_CONNECTION_STRING = "sqlite:///conversations.db"


class persistent_chatbot:
    def __init__(self, session_id : Optional[str] = None):
        """Initialize the Chatbot."""
        self.model=init_chat_model(model_provider=MODEL_PROVIDER,model=DEFAULT_MODEL, temperature=0)
        self.system_msg=SystemMessage(content=SM)
        self.messages=[self.system_msg]
        self.initialise_session(session_id)
        self.initialise_history()

    def initialise_session(self, session_id : Optional[str] = None) -> None:
        if session_id == None:
            self.session_id = self.generate_session_id()
        else:
            self.session_id = session_id

    def generate_session_id(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Chat - {timestamp}"

    def initialise_history(self) -> None:
        """Initialize or load the conversation for the current session."""
        self.history =SQLChatMessageHistory(
            session_id=self.session_id, connection=DB_CONNECTION_STRING
        )

        if not self.history.get_messages():
            self.history.add_message(self.system_msg)


if __name__ == "__main__":
    result=persistent_chatbot()
    print(result.initialise_history())
