import os

# ========== Define Directory Structure ==========
DIR_STRUCTURE = [
    "backend",
    "backend/core",
    "backend/services",
    "backend/api",
    "backend/api/routes",
    "backend/models",
    "backend/database",
    "backend/tests",
    "frontend"  # placeholder for frontend setup
]

# ========== Define Files and Initial Content ==========
FILE_CONTENTS = {
    "backend/main.py": """\
from fastapi import FastAPI
from backend.api.routes import search, qa, summary, auth

app = FastAPI(title="Academic Paper RAG System")

# Include routers
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(qa.router, prefix="/qa", tags=["Q&A"])
app.include_router(summary.router, prefix="/summary", tags=["Summary"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
""",

    "backend/core/__init__.py": "",
    "backend/core/config.py": """\
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
""",

    "backend/core/logger.py": """\
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
""",

    "backend/services/arxiv_service.py": """\
# Handles ArXiv paper search
def search_arxiv(query: str, max_results: int = 10):
    pass
""",

    "backend/services/openalex_service.py": """\
# Gets citation count from OpenAlex
def get_citation_count(title: str) -> int:
    pass
""",

    "backend/services/pdf_service.py": """\
# Downloads PDF and extracts text
def pdf_to_text(pdf_url: str) -> str:
    pass
""",

    "backend/services/embedding_service.py": """\
# Chunking and embedding using Sentence Transformers
def chunk_and_embed(text: str):
    pass
""",

    "backend/services/rag_service.py": """\
# RAG Chain using Mistral API
def run_qa(vectorstore, question: str) -> str:
    pass
""",

    "backend/services/suggestion_service.py": """\
# Generate suggested questions using LLM
def suggest_questions(context: str):
    pass
""",

    "backend/api/routes/__init__.py": "",
    "backend/api/routes/search.py": """\
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def search_papers(query: str):
    return {"message": f"Search papers for: {query}"}
""",

    "backend/api/routes/qa.py": """\
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def ask_question(question: str):
    return {"answer": f"Answer for: {question}"}
""",

    "backend/api/routes/summary.py": """\
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def summarize(paper_id: str):
    return {"summary": f"Summary of paper: {paper_id}"}
""",

    "backend/api/routes/auth.py": """\
from fastapi import APIRouter

router = APIRouter()

@router.post("/login")
def login():
    return {"token": "fake-jwt"}

@router.post("/register")
def register():
    return {"message": "User registered"}
""",

    "backend/models/__init__.py": "",
    "backend/database/__init__.py": "",
    "backend/tests/__init__.py": "",
    "frontend/README.md": "# Frontend will go here",
    ".env": "MISTRAL_API_KEY=your_key_here\n"
}

# ========== Scaffolding Function ==========
def scaffold_project():
    for folder in DIR_STRUCTURE:
        os.makedirs(folder, exist_ok=True)
        init_file = os.path.join(folder, "__init__.py")
        if "tests" not in folder and "frontend" not in folder:
            with open(init_file, "w") as f:
                f.write("")

    for filepath, content in FILE_CONTENTS.items():
        with open(filepath, "w") as f:
            f.write(content)

    print("✅ Project structure created successfully!")

# ========== Run ==========
if __name__ == "__main__":
    scaffold_project()
