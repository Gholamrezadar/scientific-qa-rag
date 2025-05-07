from typing import List

import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings


from .models import EMBEDDING_MODEL_NAME
print("Initializing Chroma DB...")
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
client_settings = Settings(
    is_persistent=True,
    anonymized_telemetry=False  # Disable telemetry
)
vector_db = Chroma(collection_name='rag-db', persist_directory='rag_db', embedding_function=embedding_model, client_settings=client_settings)

def load_documents(doc_dir: str) -> List[str]:
    '''Loads the documents from the specified directory.

    Args:
        doc_dir (str): The directory containing the documents.

    Returns:
        List[str]: The documents.
    '''

    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            path = os.path.join(doc_dir, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                documents.append(doc.page_content)
    return documents


def chunk_document(document: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    '''Divides the document into smaller chunks using langchain RecursiveTextSplitter.

    Args:
        document (str): The document to chunk.
        chunk_size (int, optional): The size of each chunk.
        chunk_overlap (int, optional): The overlap between chunks.

    Returns:
        List[str]: The chunked documents.
    '''

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(document)


# def embed_chunks(chunks: List[str]) -> List[List[float]]:
#     '''Embeds the chunks using the specified embedding model.

#     Args:
#         chunks (List[str]): The chunked documents.

#     Returns:
#         List[List[float]]: The embedded chunks.
#     '''
#     embedded_chunks = []
#     for chunk in chunks:
#         embedded_chunks.append(embedding_model.embed_query(chunk))
#     return embedded_chunks


# def embed_question(question: str) -> List[float]:
#     '''Embeds the question using the specified embedding model.

#     Args:
#         question (str): The question to embed.

#     Returns:
#         List[float]: The embedded question.
#     '''

#     return embedding_model.embed_query(question)


def store_chunks_in_db(chunks: List[str]) -> None:
    '''Stores chunks in the database.

    Args:
        chunks (List[str]): The chunks to store.
    '''
    vector_db.add_texts(texts=chunks)

    
def retrieve_relevant_chunks(question: str, k: int = 2) -> List[str]:
    '''Retrieves the relevant chunks from the database using the specified embedding model.

    Args:
        question (str): The question to embed.
        k (int, optional): The number of chunks to retrieve.

    Returns:
        List[str]: The relevant chunks.
    '''
    retrieved_results = vector_db.similarity_search(query=question, k=k)
    relevant_chunks: List[str] = []
    for result in retrieved_results:
        relevant_chunks.append(result.page_content)
    return relevant_chunks

