import streamlit as st

import database
import time

def delete_db_button():
        db_get = st.button("Reset Database")

        if db_get:
            database.reset_database()

def display_list_of_documents():
    st.subheader("Documents available")
    documents_names = database.get_document_names()
    if documents_names:
        for doc_name in documents_names:
            col_doc_name, col_doc_include, col_del_button = st.columns([4,3,1]) #Collums layout
            
            with col_doc_name:
                st.write(doc_name)

            with col_doc_include:
                if f"toggle_{doc_name}" not in st.session_state:
                    st.session_state[f"toggle_{doc_name}"] = True #padrão é 1 -> incluso na busca
                st.checkbox("Included in search", key=f"toggle_{doc_name}")

            with col_del_button:
                delete_doc_button = st.button("X", key=f"delete_{doc_name}" )
                if delete_doc_button:
                    with st.spinner(f"Deleting {doc_name}"):
                        is_success = database.remove_document_from_db(doc_name)
                        if is_success:
                            st.toast(f"Document '{doc_name}' deleted successfully!")



def sidebar():


    with st.sidebar:

        st.header("Rag Question Answer")

        #Verifica se já processou os arquivos
        processed_in_session = st.session_state.get('files_processed', False)

        uploaded_file= st.file_uploader("Upload PDF File for QnA", type=["pdf"], accept_multiple_files=True, disabled = processed_in_session)


        process = st.button(
            "Process"
        )

        if uploaded_file and process:
            with st.spinner("Inserting the documents in the database...", show_time = True):
                for doc in uploaded_file:
                    normalize_uploaded_file_name = doc.name.translate(
                        str.maketrans({"-":"_", ".": "_", " ":"_"})
                    )
                    all_splits = database.process_document(doc)
                    database.add_to_vector_collection(all_splits, normalize_uploaded_file_name, doc.name)
            st.session_state.files_processed = True
            st.success("Documents processed successfully!")
            st.toast("Documents ready! Click 'Generate Draft' to start")
            time.sleep(1)

        if processed_in_session and st.button("Upload new Documents"):
            st.session_state.files_processed = False
            st.rerun()

        display_list_of_documents()

