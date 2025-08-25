import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from createDb import generate_data_store
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


CHROMA_PATH = "./chroma"

def main():
## Load Documents, Split, Store, embed
    generate_data_store()
    #Retrieve 
    EMBEDDING_FUNCTION = OpenAIEmbeddings(model = "text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_FUNCTION)

    query_text = "What is Langmuir Blodgett Assembly?"

    #Documento e sua relevancia
    #List[Tuple[Document,float]]
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0:
        print(f"Unable to find matching results.")
        return 

    print(f"Similaridade dos 3 primeiros: {results[0][1]} - {results[1][1]} - {results[2][1]}")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)

    #Generate
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    ---

    Do not use previous trained data.
    If the information is not suficient, say so.
    
    Answer this question: {question} 
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)
    print(prompt)


    model = OllamaLLM(model= "llama3.2")
    response_text = model.invoke(prompt)
    print(response_text)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response {response_text}\nSources: {sources}"
    # print(formatted_response)

if __name__ == "__main__":
    main()