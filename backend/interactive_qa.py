import logging
from backend.services.arxiv_service import search_arxiv, download_pdf
from backend.services.openalex_service import get_citation_count
from backend.services.pdf_service import extract_clean_text_from_pdf
from backend.services.embedding_service import embed_chunks ,chunk_text
from backend.services.rag_service import evaluate_rag  
from backend.core.config import Settings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def start_qa_chain(vectorstore):
    llm = ChatOpenAI(
        api_key=Settings.MISTRAL_API_KEY,
        base_url=Settings.MISTRAL_API_BASE,
        model=Settings.DEFAULT_MODEL,
        temperature=0.3
        
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,  # ✅ So we can evaluate based on sources
    )

    print("\n🤖 Ready for Q&A. Type 'exit' to quit.")
    while True:
        question = input("Ask a question : ")
        if question.lower() == "exit":
            break
        try:
            result = qa({"query": question})
            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            print(f"\n📖 Answer: {answer}")

            # ✅ Evaluate RAG output
            evaluation = evaluate_rag(answer, question, sources)
            print("📊 Evaluation Metrics:")
            print(f"   Relevance score       : {evaluation['relevance']}")
            print(f"   Supporting documents  : {evaluation['supporting_docs']}")
            print(f"   Hallucination score   : {evaluation['hallucination_score']}\n")

        except Exception as e:
            print(f"❌ Error during answering: {e}")


def main():
    query = input(" RESEARCH PAPER FINDER : ")
    papers = search_arxiv(query, max_results=10)

    if not papers:
        print("❌ No papers found.")
        return

    print(f"\n📚 Found {len(papers)} papers. Fetching citation counts...")
    for paper in papers:
        paper["citations"] = get_citation_count(paper["title"])
    papers.sort(key=lambda x: x["citations"], reverse=True)

    print("\nTop Papers:")
    for i, p in enumerate(papers):
        print(f"{i+1}. {p['title']} ({p['citations']} citations)")

    selected_title = input("\n✏️ Enter the exact title of the paper you want to chat with: ").strip()
    selected_paper = next((p for p in papers if p["title"].lower() == selected_title.lower()), None)

    if not selected_paper:
        print("❌ Paper not found.")
        return

    # Where you call the function:
    arxiv_id = selected_paper["pdf_url"].split("/pdf/")[-1].replace(".pdf", "")
    local_path = download_pdf(arxiv_id, selected_title)

    if not local_path:
        print("❌ Failed to download PDF.")
        return

    text = extract_clean_text_from_pdf(local_path)
    if not text:
        print("❌ Failed to extract text from PDF.")
        return

    print("📚 Chunking and embedding...")
    chunks = chunk_text(text)
    vectorstore = embed_chunks(chunks)

    start_qa_chain(vectorstore)


if __name__ == "__main__":
    main()
