import streamlit as st

import ollama
import database
from sidebar import sidebar

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


def call_llm(context: str, prompt:str):
    #LLM_MODEL = "llama3.2:3b"
    LLM_MODEL = "deepseek-r1"

    response = ollama.chat(
        model= LLM_MODEL,
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
    sidebar()

    st.header("RAG Question Answer")

    prompt = st.text_area(" Ask a question related to your document: ")
    ask = st.button("Ask")

    if ask and prompt:
        with st.spinner("Looking for answers...", show_time= True):
            
            excluded_docs= [
                doc_name for doc_name in database.get_document_names()
                if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
            ]
            most_similar_docs = database.query_collection(prompt, exclude_docs=excluded_docs)
            query_embed = database.query_collection(prompt) #N sei se é o melhor método pra fazer embedding do prompt
            context = query_embed.get("documents")[0]
            response = call_llm(most_similar_docs.items, context)
            st.write_stream(response)
        
            with st.expander("See retrivied documents"):
                st.write(most_similar_docs)




if __name__ == "__main__":
    main()