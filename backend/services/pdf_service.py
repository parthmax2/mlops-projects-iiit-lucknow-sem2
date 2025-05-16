# backend/services/pdf_parser_service.py

import os
import fitz  # PyMuPDF
import re
from typing import Optional


def extract_clean_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts and cleans text from a PDF file.
    Returns None if extraction fails.
    """
    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
        raw_text = "\n".join([page.get_text() for page in doc])
        return clean_pdf_text(raw_text)
    except:
        return None


def clean_pdf_text(text: str) -> str:
    """
    Cleans PDF text for downstream NLP use.
    Removes artifacts, headers, citations, and LaTeX remnants.
    """
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"arXiv:\d+\.\d+(v\d+)?\s+\[.*?\]\s+\d{4}", "", text)
    text = re.sub(r"\n?\s*Page \d+\s*\n?", "\n", text)
    text = re.sub(r"\n\s*[A-Z][^\n]{0,60}\s+\d{1,3}\s*\n", "\n", text)
    text = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", text)
    text = re.sub(r"\\(begin|end|cite|ref|label|frac|textbf|emph|section|subsection|item|eqref)\{[^}]*\}", "", text)
    text = re.sub(r"^\s*[\(\[]?[A-Za-z0-9\s=+\-*/^<>\\]{1,50}[\)\]]?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join([line for line in text.split("\n") if len(line.strip()) > 5])
    cleaned = text.strip()
    return cleaned
