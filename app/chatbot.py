from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from app.prompts import SYSTEM_MESSAGE as SM



# Load environment variables from .env
load_dotenv()
#print(bool(os.getenv("clinical_AI")) ) 
MODEL_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class ChatBot:
    def __init__(self):
        """Initialize the Chatbot."""
        self.model = init_chat_model(model_provider=MODEL_PROVIDER,model=DEFAULT_MODEL, temperature=0)
        self.system_msg = SystemMessage(content=SM)
        self.messages = [self.system_msg]


    def get_response(self, user_message: str) -> str:
        """Process user message and get AI response."""
        # Receive the user's message and append it to the conversation
        self.messages.append(HumanMessage(content=user_message))

        # Send the conversation to the model
        response = self.model.invoke(self.messages)

        # Return the conversation to the user
        self.messages.append(AIMessage(content=response.text))

        return response.text

if __name__ == "__main__":

    user_message="What is the difference between an Agent and a Conversational bot?"
    result=ChatBot().get_response(user_message)
    print(result)