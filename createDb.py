from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "pdfs"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True,
    )

    print(f"Split {len(docs)} documents into {len(chunks)} chunks")

    doc = chunks[10]

    print("Document content: ",doc.page_content)
    print("Document metadata : ", doc.metadata)

    chunks = text_splitter.split_documents(docs)
    return chunks


def save_to_chroma(chunks):
    #Clear out the existing database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #Create db from docs
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory= CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()