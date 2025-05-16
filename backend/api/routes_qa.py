import logging
import os

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.services.pdf_service import extract_clean_text_from_pdf
from backend.services.embedding_service import embed_chunks , chunk_text
from backend.services.rag_service import evaluate_rag
from backend.core.config import Settings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")

QA_PROMPT_TEMPLATE = """
You are an expert assistant tasked with extracting accurate answers strictly from the provided document sections.

Instructions:
- Use ONLY the information in the given context.
- Do NOT rely on prior knowledge or make assumptions.
- Cite or reference relevant phrases from the context if helpful.
- If the answer is not present, reply exactly: "The answer is not found in the provided sections."

Context:
{context}

Question:
{question}
"""

qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

@router.get("/qa", response_class=HTMLResponse)
async def qa_form(request: Request):
    return templates.TemplateResponse("qa_form.html", {"request": request})


@router.post("/qa", response_class=HTMLResponse)
async def qa_submit(request: Request, title: str = Form(...), question: str = Form(...)):
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
    pdf_path = os.path.join("assets", "papers", f"{safe_title}.pdf")

    if not os.path.exists(pdf_path):
        return templates.TemplateResponse("qa_form.html", {
            "request": request,
            "error": "PDF not found. Please select a paper first."
        })

    try:
        text = extract_clean_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        vectorstore = embed_chunks(chunks)

        llm = ChatOpenAI(
            api_key=Settings.MISTRAL_API_KEY,
            base_url=Settings.MISTRAL_API_BASE,
            model=Settings.DEFAULT_MODEL,
            temperature=0.3,
            max_tokens=1024
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        result = qa({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        evaluation = evaluate_rag(answer, question, sources)

        return templates.TemplateResponse("qa_result.html", {
            "request": request,
            "title": title,
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": [doc.metadata.get("source", "") for doc in sources]
        })

    except Exception as e:
        logger.exception("❌ Error during QA pipeline:")
        return templates.TemplateResponse("qa_form.html", {
            "request": request,
            "error": f"Processing error: {str(e)}"
        })
