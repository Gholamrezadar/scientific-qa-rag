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


def chunk_document(document: str):
    '''Divides the document into smaller chunks using langchain RecursiveTextSplitter.

    Args:
        document (str): The document to chunk.

    Returns:
        List[str]: The chunked documents.
    '''

    pass


def embed_chunks(chunks: List[str]) -> List[str]:
    '''Embeds the chunks using the specified embedding model.

    Args:
        chunks (List[str]): The chunked documents.

    Returns:
        List[str]: The embedded chunks.
    '''

    pass


def embed_question(question: str) -> str:
    '''Embeds the question using the specified embedding model.

    Args:
        question (str): The question to embed.

    Returns:
        str: The embedded question.
    '''

    pass


def store_vector_in_db(vectors):
    '''Stores an embedded vector in a database.

    Args:
        vectors (List[str]): The vectors to store.
    '''

    pass


def retrieve_relevant_chunks(embeded_question: str) -> List[str]:
    '''Retrieves the relevant chunks from the database using the specified embedding model.

    Args:
        embeded_question (str): The embedded question.

    Returns:
        List[str]: The relevant chunks.
    '''

    pass
