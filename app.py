from typing import List
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader

import os
import tempfile
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama

SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

### Database ####
def add_to_vector_collection(all_splits:list[Document], filename: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [],[],[]
    st.success("Adding to vector Storage!")


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
### Database ####


def call_llm(context: str, prompt:str):
    response = ollama.chat(
        model= "llama3.2:3b",
        stream = True,
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            }
        
        ]
    )

    #Como está no modo stream, a resposa virá por chunks
    #O último chunk virá com a mensagem "done"
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break




def main():

    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        st.header("Rag Question Answer")
        uploaded_file= st.file_uploader("Upload PDF File for QnA", type=["pdf"], accept_multiple_files=False)

        process = st.button(
            "Process"
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-":"_", ".": "_", " ":"_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    st.header("RAG Question Answer")

    prompt = st.text_area(" Ask a question related to your document: ")
    ask = st.button("Ask")

    if ask and prompt:
        most_similar_docs = query_collection(prompt)
        #st.write(most_similar_docs)
        query_embed = query_collection(prompt)
        context = query_embed.get("documents")[0]
        response = call_llm(most_similar_docs.items, context)
        st.write_stream(response)
    
        with st.expander("See retrivied documents"):
            st.write(most_similar_docs)




if __name__ == "__main__":
    main()