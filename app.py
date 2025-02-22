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
    sidebar()

    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "ask" not in st.session_state:
        st.session_state.ask= False

    st.header("RAG Question Answer")

    prompt = st.text_area(" Ask a question related to your document: ", value = st.session_state.prompt)
    ask = st.button("Ask")

    if ask: 
        st.session_state.prompt = prompt
        st.session_state.ask = True

    print("sesssion prompt: ", st.session_state.prompt)
    print("session state ask: ", st.session_state.ask)

    
    if st.session_state.ask and st.session_state.prompt:
        with st.spinner("Looking for answers...", show_time= True):
            most_similar_docs = database.query_collection(prompt)
            query_embed = database.query_collection(prompt)
            context = query_embed.get("documents")[0]
            response = call_llm(most_similar_docs.items, context)

            if "llm_answer" not in st.session_state:
                st.session_state.llm_answer = ""

            st.write_stream(response)
            
            st.session_state.llm_answer = response
        
            with st.expander("See retrivied documents"):
                st.write(most_similar_docs)




if __name__ == "__main__":
    main()