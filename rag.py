import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader

from langchain_community.vectorstores import Chroma

DATA_PATH = "./pdfs"
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents



## Load Documents ##

file_path = ( "./pdfs/2.pdf")
loader = PyPDFLoader(file_path, extract_images=True)
docs = []
docs_lazy = loader.load()

for doc in docs_lazy:
    docs.append(doc)

# print(docs[1].page_content)
# print(docs[1].metadata)

##Split

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len, add_start_index = True)
all_splits = text_splitter.split_documents(docs)

chunks = text_splitter.split_documents(docs)

print(chunks)

##Embed

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#Store

CHROMA_PATH = "chroma"

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

#remover se já existe
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory = CHROMA_PATH)
#Retrieve 

#Generate

# if not os.environ.get("OPEN_API_KEY"):
#     os.environ["OPEN_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# model = ChatOpenAI(model="gpt-4o-mini")

# print(model.invoke("Hello, World"))
