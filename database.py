import chromadb
from langchain_core.documents import Document
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import tempfile
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def add_to_vector_collection(all_splits:list[Document], filename: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [],[],[]

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{filename}_{idx}")

    collection.upsert(
        documents= documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def get_vector_collection() -> chromadb.Collection:
    #Able to use ollama as api embedding function
    ollama_ef = OllamaEmbeddingFunction(
        url = "http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function= ollama_ef,
        metadata= {"hnsw:space": "cosine"}, #Calculo de similaridade
    )

def process_document(uploaded_file: UploadedFile) -> List[Document]: 
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name) #Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 100,
        separators= ["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    return text_splitter.split_documents(docs)

def query_collection(prompt:str, n_results: int = 5):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results= n_results)

    return results