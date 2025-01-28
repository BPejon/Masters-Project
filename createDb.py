from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")
CHROMA_PATH = "chroma"
DATA_PATH = "pdfs"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(documents)
    print("------")
    return documents

def split_documents(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Split {len(docs)} documents into {len(chunks)} chunks")

    # doc = chunks[10]

    # print("Document content: ",doc.page_content)
    # print("Document metadata : ", doc.metadata)

    return chunks


def save_to_chroma(chunks):
    #Clear out the existing database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #Create db from docs
    db = Chroma.from_documents(
        chunks,embedding= EMBEDDING_FUNCTION, persist_directory= CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()