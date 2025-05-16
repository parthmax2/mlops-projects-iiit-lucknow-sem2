import sys
import os

# Ensure the backend directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.arxiv_service import search_arxiv
from services.openalex_service import get_citation_count
import logging

def test_arxiv_openalex_combined():
    query = "transformer language model"
    max_results = 5

    print(f"\n Testing ArXiv + OpenAlex integration for query: '{query}'")
    papers = search_arxiv(query, max_results=max_results)

    if not papers:
        print(" No papers found from ArXiv.")
        return

    print(f"\n Found {len(papers)} papers. Fetching citations from OpenAlex...\n")

    for i, paper in enumerate(papers):
        title = paper["title"]
        citations = get_citation_count(title)
        print(f"{i+1}. {title} — Citations: {citations if citations is not None else 'N/A'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_arxiv_openalex_combined()
