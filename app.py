import streamlit as st
from rag_pipeline import create_rag_chain

st.set_page_config(page_title="ChatGroq RAG App")

st.title("📚 ChatGroq RAG Document Chatbot")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf","txt"]
)

if uploaded_file:

    with open("temp_doc", "wb") as f:
        f.write(uploaded_file.read())

    rag = create_rag_chain("temp_doc")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask something about the document")

    if user_input:

        st.session_state.messages.append({
            "role":"user",
            "content":user_input
        })

        response = rag(user_input)

        st.session_state.messages.append({
            "role":"assistant",
            "content":response
        })

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])

        else:
            st.chat_message("assistant").write(msg["content"])