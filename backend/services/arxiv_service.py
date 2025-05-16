# backend/services/arxiv_service.py

import os
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_PDF_DIR = os.path.join("backend", "assets", "papers")
MIN_YEAR = 2015


def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search ArXiv for papers related to the given query,
    filtered by publication year >= 2015.
    """
    url = f"{ARXIV_API_URL}?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Error fetching ArXiv data: {response.status_code}")
        return []

    try:
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            published_date = entry.find("atom:published", ns).text[:10]
            year = int(published_date[:4])

            if year < MIN_YEAR:
                continue

            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib["href"]
                    break

            papers.append({
                "title": title,
                "summary": summary,
                "authors": [],  # No authors parsed in this version
                "arxiv_id": arxiv_id,
                "published": published_date,
                "year": year,
                "pdf_url": pdf_url
            })

        return papers

    except Exception as e:
        print("❌ Failed to parse ArXiv response.")
        raise


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace("\n", " ")[:150]


def download_pdf(arxiv_id: str, title: str, output_dir: str = DEFAULT_PDF_DIR) -> str:
    print(f"⬇️ Downloading: {title}")
    os.makedirs(output_dir, exist_ok=True)

    safe_title = sanitize_filename(title)
    pdf_filename = f"{safe_title}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    if os.path.exists(pdf_path):
        print(f"✅ PDF already exists: {pdf_path}")
        return pdf_path

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ PDF saved to: {pdf_path}")
        return pdf_path

    except requests.RequestException as e:
        print(f"❌ Error downloading PDF from: {pdf_url}")
        raise
