## config.py - settings and the LLM used by all 3 agents
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  ## reads GROQ_API_KEY from the .env file

MAX_TOOL_STEPS = 6   ## safety limit so the researcher never loops forever
MAX_RETRIES = 1      ## how many times the writer can send the work back to the researcher
MEMORY_TURNS = 4     ## how many past question-answer pairs we remember


def get_llm(temperature=0):
    model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=temperature)
