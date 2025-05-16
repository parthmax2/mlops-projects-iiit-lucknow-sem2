import logging
from typing import List, Dict, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS as CommunityFAISS
from langchain_openai import ChatOpenAI


from backend.services.embedding_service import embed_chunks
from backend.services.pdf_service import extract_clean_text_from_pdf
from backend.services.embedding_service import  chunk_text
from backend.core.config import Settings


# Logging setup
logger = logging.getLogger(__name__)

# Prompt template with document structure awareness
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

# Initialize LLM
llm = ChatOpenAI(
    api_key=Settings.MISTRAL_API_KEY,
    base_url=Settings.MISTRAL_API_BASE,
    model=Settings.DEFAULT_MODEL
)


def build_ensemble_retriever(vectorstore: FAISS, chunks: List[str]) -> EnsembleRetriever:
    """
    Combines semantic (vector) and keyword-based search using EnsembleRetriever.
    """
    logger.info("Setting up hybrid retrieval (semantic + keyword)")
    
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    keyword_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.7, 0.3]  # Adjustable weights
    )

    logger.info("Hybrid retriever configured")
    return retriever


def generate_answer(pdf_path: str, question: str) -> Optional[str]:
    """
    Complete RAG pipeline from PDF to answer generation.
    """
    logger.info(f"Processing PDF for RAG: {pdf_path}")

    # Step 1: Extract & clean PDF text
    raw_text = extract_clean_text_from_pdf(pdf_path)
    if not raw_text:
        logger.error("Failed to extract text from PDF.")
        return None

    # Step 2: Chunk the text
    chunks = chunk_text(raw_text)
    if not chunks:
        logger.error("Chunking failed.")
        return None

    # Step 3: Embed & index in vectorstore
    vectorstore = embed_chunks(chunks)
    if not vectorstore:
        logger.error("Failed to embed text and create vectorstore.")
        return None

    # Step 4: Hybrid Retrieval
    retriever = build_ensemble_retriever(vectorstore, chunks)

    # Step 5: RAG with prompt template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    logger.info("Running RetrievalQA chain...")
    result = qa_chain({"query": question})

    answer = result.get("result", "")
    sources = result.get("source_documents", [])

    logger.info("Answer generation complete.")
    return {
        "answer": answer,
        "sources": [doc.metadata.get("source", "") for doc in sources]
    }


# Optional: Quality Metrics (stub)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_rag(answer: str, query: str, sources: List[str]) -> Dict[str, float]:
    """
    Basic RAG output quality evaluation.
    """
    
    query_emb = embedding_model.encode([query])[0].reshape(1, -1)
    answer_emb = embedding_model.encode([answer])[0].reshape(1, -1)

    relevance = float(cosine_similarity(query_emb, answer_emb)[0][0])
    hallucination_score = 1.0 if "I cannot determine" in answer else 0.0

    return {
        "relevance": round(relevance, 3),
        "supporting_docs": len(sources),
        "hallucination_score": hallucination_score
    }
