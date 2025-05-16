import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Settings:
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    MISTRAL_API_BASE: str = "https://api.mistral.ai/v1"
    DEFAULT_MODEL: str = "mistral-small"

    # You can add more future configs here like:
    # DB_URL, FRONTEND_ORIGIN, DEBUG_MODE, etc.

    @classmethod
    def validate(cls):
        if not cls.MISTRAL_API_KEY:
            raise ValueError(" MISTRAL_API_KEY is missing in .env file")

# Validate on import
Settings.validate()
