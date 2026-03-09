from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from document_loader import load_documents, split_documents
from groq_chat import load_llm


# --------------------------------------------
# Create Vector Store
# --------------------------------------------

def create_vectorstore(file_path):

    # Load documents
    documents = load_documents(file_path)

    # Split into chunks
    chunks = split_documents(documents)

    # Create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# --------------------------------------------
# Prompt Template
# --------------------------------------------

def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer the question ONLY using the context below.

If the answer is not present in the context, say:
"I cannot find this information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return prompt


# --------------------------------------------
# Create RAG Chain
# --------------------------------------------

def create_rag_chain(file_path):

    # Create vector database
    vectorstore = create_vectorstore(file_path)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Load Groq LLM
    llm = load_llm()

    # Create prompt template
    prompt = create_prompt()

    # Main RAG pipeline
    def rag_pipeline(question):

        # Retrieve relevant docs
        docs = retriever.invoke(question)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Create final prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # Generate response from Groq
        response = llm.invoke(final_prompt)

        return response.content

    return rag_pipeline