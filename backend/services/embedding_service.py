import logging
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# Constants
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Splits text into overlapping chunks using RecursiveCharacterTextSplitter.
    """
    if not text:
        
        return []

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    
    return chunks


def embed_chunks(
    chunks: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> Optional[FAISS]:
    """
    Converts text chunks to vector embeddings using HuggingFace models.
    Returns a FAISS vector store.
    """
    if not chunks:
        
        return None

    try:
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        

        return vectorstore

    except Exception as e:
        
        return None
