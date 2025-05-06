from typing import List

import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from models import EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME

embedding_model = OllamaEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(collection_name='rag-db', persist_directory='rag_db', embedding_function=embedding_model)


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
            loader = TextLoader(path)
            docs = loader.load()
            for doc in docs:
                documents.append(doc.page_content)
    return documents


def chunk_document(document: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
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


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    '''Embeds the chunks using the specified embedding model.

    Args:
        chunks (List[str]): The chunked documents.

    Returns:
        List[List[float]]: The embedded chunks.
    '''
    embedded_chunks = []
    for chunk in chunks:
        embedded_chunks.append(embedding_model.embed_query(chunk))
    return embedded_chunks


def embed_question(question: str) -> List[float]:
    '''Embeds the question using the specified embedding model.

    Args:
        question (str): The question to embed.

    Returns:
        List[float]: The embedded question.
    '''

    return embedding_model.embed_query(question)


def store_vector_in_db(vectors: List[List[float]]) -> None:
    '''Stores an embedded vector in a database.

    Args:
        vectors (List[List[float]]): The vectors to store.
    '''

    vector_db.add(vectors)
    vector_db.persist()



def retrieve_relevant_chunks(embedded_question: List[float], k: int = 2) -> List[str]:
    '''Retrieves the relevant chunks from the database using the specified embedding model.

    Args:
        embedded_question (List[float]): The embedded question.

    Returns:
        List[str]: The relevant chunks.
    '''

    results = vector_db.similarity_search_by_vector(embedded_question, k=k)
    return [doc.page_content for doc in results]
