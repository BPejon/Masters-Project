import streamlit as st

import ollama
import database
from sidebar import sidebar


SYSTEM_PROMPT :str= """
You are a PDH Professor focused on Material Science papers.
Your task is to write an outline of a review paper on the subject given within CONTEXT.

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
2. Organize your answer into paragraphs and sections.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""
SYSTEM_PROMPT_AFTER = """

You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

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
The next line will be offered you the prompt again:
""" 

# Nao faço ideia do pq mas essa função n funciona....
def generate_text_stream(context: str, prompt:str, system_prompt:str = SYSTEM_PROMPT):
    messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            },
            {
                "role": "system",
                "content": SYSTEM_PROMPT_AFTER,
            },
                        {
                "role": "user",
                "content": f"Prompt : {prompt}" ,
            },
        ]
    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = True,
        messages = messages
    )

    buffer = ""  # Buffer para acumular conteúdo entre chunks
    in_think_block = False  # Flag para indicar se estamos dentro de um bloco <think>

    for chunk in response:
        if chunk["done"] is False:
            content = chunk["message"]["content"]
            buffer += content
            
            # Processa o buffer para remover blocos <think>
            while True:
                if not in_think_block:
                    # Procura por abertura de <think>
                    think_start = buffer.find("<think>")
                    if think_start == -1:
                        # Não há <think>, pode yield todo o buffer
                        if buffer:
                            yield buffer
                            buffer = ""
                        break
                    
                    # Yield conteúdo antes do <think>
                    if think_start > 0:
                        yield buffer[:think_start]
                    
                    # Atualiza buffer e flag
                    buffer = buffer[think_start + len("<think>"):]
                    in_think_block = True
                else:
                    # Procura por fechamento de </think>
                    think_end = buffer.find("</think>")
                    if think_end == -1:
                        # Não encontrou fechamento, descarta o buffer atual
                        buffer = ""
                        break
                    
                    # Pula o conteúdo entre <think> e </think>
                    buffer = buffer[think_end + len("</think>"):]
                    in_think_block = False
                    
                    # Continua processando o restante do buffer
        else:
            break
    
    # Yield qualquer conteúdo restante fora de blocos <think>
    if buffer and not in_think_block:
        yield buffer

def combine_drafts(draft1: str, draft2:str, prompt:str):

    
    combine_prompt = f"""
        You are an expert in information synthesis. Your task is to combine two versions of a scientific article structure on the same topic into a single refined version.

        Topic: {prompt}

        Version 1:
        {draft1}

        Version 2:
        {draft2}

        Instructions:

            Carefully analyze both versions

            Identify the strengths of each

            Combine the best parts from each version

            Maintain a logical and cohesive structure

            Produce a single refined version that is better than both individual versions

        Return only the final refined structure, without additional comments.
    """
    messages = [
            {
                "role": "system",
                "content": "You are an assistant specialized in combining and refining scientific article structures.",
            },
            {
                "role": "user",
                "content": combine_prompt ,
            }
        ]
    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = False,
        messages = messages
    )

    #Como está no modo stream, a resposa virá por chunks
    #O último chunk virá com a mensagem "done"
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def make_one_draft(context: str, prompt:str):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            },
            {
                "role": "system",
                "content": SYSTEM_PROMPT_AFTER,
            },
                        {
                "role": "user",
                "content": f"Prompt : {prompt}" ,
            },
        ]

    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = True,
        messages = messages
    )

def refine_full_article(all_sections: list[str], user_prompt: str):
    """Envia todas as seções para a LLM revisar e tirar inconsistências"""
    full_text = "\n\n".join(all_sections)

    refine_prompt = f"""
    You are a professor in Material Science. You received multiple sections of a scientific review paper.
    Your task is to unify them into a single coherent article.

    Topic: {user_prompt}

    Sections draft:
    {full_text}

    Instructions:
    - Eliminate inconsistencies and redundancies
    - Ensure logical flow between sections
    - Improve transitions
    - Keep scientific tone
    - Return only the final unified text
    """

    response = ollama.chat(
        model=st.session_state.llm_model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are an expert scientific editor."},
            {"role": "user", "content": refine_prompt}
        ]
    )
    return response["message"]["content"]

def generate_text_llm_no_stream(context: str, prompt:str, system_prompt:str = SYSTEM_PROMPT):
    messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}" ,
            },
        ]

    response = ollama.chat(
        model= st.session_state.llm_model,
        stream = False,
        messages = messages
    )
    full_response = response['message']['content']
    # Extrai o conteúdo <think> se existir
    # Extrai TODO o conteúdo <think> (mesmo que mal formado ou múltiplos)
    think_contents = []
    while "<think>" in full_response:
        start = full_response.find("<think>") + len("<think>")
        end = full_response.find("</think>", start) if "</think>" in full_response[start:] else len(full_response)
        
        think_content = full_response[start:end].strip()
        think_contents.append(think_content)
        
        # Remove o bloco atual (com ou sem fechamento)
        full_response = full_response[:start-len("<think>")] + full_response[end+(len("</think>") if "</think>" in full_response[start:] else 0):]
    
    # Armazena todos os conteúdos <think> encontrados
    st.session_state.think_content = "\n\n---\n\n".join(think_contents) if think_contents else ""
    
    # Remove quaisquer fragmentos remanescentes
    full_response = full_response.replace("<think>", "").replace("</think>", "").strip()
   
    return full_response


def call_two_drafts(context: str, prompt:str):
    
    draft1= generate_text_stream(context, prompt)

    draft2= generate_text_stream(context, prompt)

    
    combined = combine_drafts(draft1, draft2, prompt)
    
    return combined



def get_most_similar_docs(prompt:str, n_chunks: int= 10, max_chuks_per_docs:int = 15):
    excluded_docs= [
        doc_name for doc_name in database.get_document_names()
        if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
    ]

    most_similar_docs = database.query_collection(prompt,n_chunks,excluded_docs, max_chuks_per_docs)
    
    return most_similar_docs

def generate_chat(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
    
        with st.spinner("Looking for answers...", show_time= True):
            
            excluded_docs= [
                doc_name for doc_name in database.get_document_names()
                if f"toggle_{doc_name}" in st.session_state and not st.session_state[f"toggle_{doc_name}"]
            ]
            most_similar_docs = database.query_collection(prompt, exclude_docs=excluded_docs)


            stream = make_one_draft(most_similar_docs["documents"], prompt)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})


        
            with st.expander("See retrieved documents"):
                st.write(most_similar_docs)
            with st.expander("See Prompt sent to LLM"):
                messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": f"Context: {most_similar_docs['documents']}, Question: {prompt}" ,
                        },
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_AFTER,
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}" ,
                        },
                ]
                st.write(messages)


def generate_sections(user_prompt:str):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # garantir lista limpa
    st.session_state.sections_drafts = []

    section_prompt= f"""
    Analyze the documents I will provide and then create 8 sections for a scientific literature review on this theme: {user_prompt}.
        
        Return only the list of sections in this exact format:
        1 - Introduction
        2 - Section Name
        ...
        8 - Conclusion
        
        Do not include any additional text or explanations.

    """
    most_similar_docs = get_most_similar_docs(section_prompt)

    sections_response = generate_text_llm_no_stream(most_similar_docs["documents"], section_prompt, "Do not put the <think> on the result text ") #Vazio pro system prompt, pq quero que o prompt acima se sobressaia
    
    print(f"Sections antes de filtrar {sections_response}")
    # Extrai as seções da resposta
    sections = [line.strip() for line in sections_response.split('\n') if line.strip()]
    
    print(f"Sections {sections}")
    
    with st.chat_message("user"):
            st.markdown(user_prompt)
    print(f"Sections antes de entrar : {sections}")

   
    # Gerar os drafts de todas as seções e salvar
    for section_theme in sections:
        draft_prompt = f"""
        Write only one section for a literature scientific review on {user_prompt} about the section {section_theme}.
        The section should be:
            - Comprehensive and detailed
            - Well-structured with paragraphs
            - Organize your answer into paragraphs and subsections.
            - Based strictly on the provided context
            - Maximum of 250 words
        """
        most_similar_docs_section_theme = get_most_similar_docs(draft_prompt, 10, 5)

        draft_response = generate_text_llm_no_stream(
            most_similar_docs_section_theme["documents"], draft_prompt
        )

        # salvar draft da seção
        st.session_state.sections_drafts.append(f"{section_theme}\n{draft_response}")

    final_article = refine_full_article(st.session_state.sections_drafts, user_prompt)

    with st.chat_message("assistant"):
        st.markdown(final_article)
    st.session_state.messages.append({"role": "assistant", "content": final_article})
            
def show_chat_interface():
    st.header("RAG Question Answer")


    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.first_interaction == True:
        st.session_state.first_interaction = False


    with st.spinner("Generating sections and first draft..."):
        generate_sections(st.session_state.research_topic)

        initial_prompt = f"""
Generate an outline of a review paper on the subject {st.session_state.research_topic}.
Use the understanding provided in the PDFs and the chunks presented in the prompt as Context.
I want the review to be comprehensive and also provide details about the methods.
I will later ask you to expand the context of the sections in the outline.
"""

        #generate_chat(initial_prompt)


    if prompt := st.chat_input("Ask a question related to your document"):
        generate_chat(prompt)

   

def show_welcome_screen():
    st.header("Welcome to RAG Question Answer")
    st.markdown(""" 
        #### Explore the literature to identify the important topics and relevant contents for a review paper on your subject.
        
        ##### How to use
        1. Upload your PDF documents using the sidebar on the left
        2. Click "Add to Database" button to add the documents into database
        3. Wait for the documents to be processed. You will see a confirmation message
        4. Specify the research topic for the LLM to create a scientific draft
        5. Select your LLM. Choose Llama3.2 if you have a low spec machine. Otherwise, select deepseek for better results.
        6. Click "Generate Draft" to generate your first draft
                
        You can Toggle documents on/off to include/exclude them from searches
""")

    document_names = database.get_document_names()

    research_topic = st.text_input(
        "Enter your research theme:",
        placeholder = "e.g Advanced Materials for Solar Cells",
        help = "This will be used to customize your scientific draft"

    )
    
    llm_model = st.radio(
        "Choose one Large Language Model to generate the Scientific Draft.",
        ["llama3.2:3b","deepseek-r1"],
        index= 0,
    )

    generate_button = st.button("Generate", disabled = not bool(document_names), help ="Upload documents to generate draft" if not document_names else "Click to generate", key = "generate_button")

    if generate_button and research_topic != "": 
        st.session_state.research_topic = research_topic
        st.session_state.llm_model = llm_model
        st.session_state.show_chat = True
        st.rerun()

def main():
    st.set_page_config(page_title="RAG Question Answer", initial_sidebar_state="expanded")

    ##Inicializa as variáveis de sessões
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "first_interaction" not in st.session_state:
        st.session_state.first_interaction = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3.2:3b"
    if "research_topic" not in st.session_state:
        st.session_state.research_topic = ""
    sidebar()

    if st.session_state.show_chat == False:
        show_welcome_screen()
    else:
        show_chat_interface()


if __name__ == "__main__":
    main()