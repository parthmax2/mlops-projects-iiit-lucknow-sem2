import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is not set in environment variables or .env file")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in environment variables or .env file")


