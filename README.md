# ChatGroq RAG Application with Streamlit UI

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using **Groq's high-speed LLM inference**, **LangChain**, and **Streamlit**. The application allows users to upload documents and ask questions about their content.

Instead of relying only on the language model’s internal knowledge, the system retrieves relevant information from uploaded documents and provides answers based strictly on that context. This improves **accuracy**, **reliability**, and **reduces hallucinations**.

The chatbot is designed as a **document question-answering system** where users can interact with their files through a conversational interface.

---

## Objectives

The main objectives of this assignment are:

* Build a **RAG pipeline** using LangChain
* Use **Groq LLM for ultra-fast inference**
* Store document embeddings in a **vector database**
* Develop an **interactive Streamlit chatbot interface**
* Ensure answers are **grounded in uploaded documents**

---

## Technologies Used

| Technology             | Purpose                               |
| ---------------------- | ------------------------------------- |
| Python                 | Core programming language             |
| LangChain              | RAG pipeline and orchestration        |
| Groq API               | High-speed LLM inference              |
| HuggingFace Embeddings | Convert text into vector embeddings   |
| FAISS                  | Vector database for similarity search |
| Streamlit              | Interactive chatbot UI                |
| Python-dotenv          | Secure API key management             |

---

## System Architecture

User Question
↓
Retriever (FAISS Vector Store)
↓
Relevant Document Chunks
↓
Prompt Template Injection
↓
Groq LLM Generation
↓
Final Answer Displayed in Streamlit

The system retrieves the most relevant document chunks and injects them into the prompt before sending it to the language model.

---

## Project Folder Structure

```
chatgroq-rag-streamlit/
│
├── app.py
├── rag_pipeline.py
├── document_loader.py
├── groq_chat.py
├── test_groq.py
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   └── sample_document.txt
│
└── screenshots/
    ├── 01_streamlit_homepage_ui.png
    ├── 02_document_uploaded.png
    ├── 03_rag_question_answer.png
    └── 04_out_of_context_question.png
```

---

## Implementation Details

### 1. Document Loading

Documents are loaded using LangChain loaders. The system supports both **TXT and PDF files**.

### 2. Text Chunking

Large documents are split into smaller chunks using:

RecursiveCharacterTextSplitter

This ensures the document can be processed efficiently within LLM context limits.

Parameters used:

* Chunk Size: 1000
* Chunk Overlap: 200

---

### 3. Embeddings Generation

Each document chunk is converted into vector embeddings using the **HuggingFace embedding model**:

sentence-transformers/all-MiniLM-L6-v2

These embeddings allow semantic similarity search.

---

### 4. Vector Database

The embeddings are stored in **FAISS**, which enables fast similarity search when retrieving relevant document chunks.

---

### 5. Retrieval-Augmented Generation

When a user asks a question:

1. The retriever searches FAISS for the most relevant chunks.
2. Retrieved context is injected into the prompt.
3. The Groq LLM generates the final answer based on that context.

This ensures responses are **grounded in the provided documents**.

---

### 6. Streamlit User Interface

The Streamlit UI includes:

* Document file uploader
* Chat input box
* Chat history display
* AI response rendering

Users can upload a document and interact with it through a conversational interface.

---

## How to Run the Project

### 1. Clone the Repository

```
git clone <repository-link>
cd chatgroq-rag-streamlit
```

---

### 2. Create Virtual Environment

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Configure API Key

Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the Application

```
python -m streamlit run app.py
```

The application will run at:

```
http://localhost:8501
```

---

## Example Questions

Users can test the chatbot with questions such as:

* What is RAG?
* What are the components of RAG?
* Why does RAG reduce hallucinations?

Example Out-of-Context Question:

* Who is the president of the USA?

The system should respond that the information is **not found in the document**.

---

## Screenshots

### Streamlit Homepage

Shows the chatbot interface and document upload option.

### Document Uploaded

Displays the uploaded document within the UI.

### RAG Question Answer

Demonstrates the chatbot answering questions based on the uploaded document.

### Out-of-Context Question

Shows the system refusing to answer when the information is not present in the document.

---

## Observations and Insights

### Why Groq is Suitable for RAG Chatbots

Groq provides extremely fast inference speeds, making it ideal for real-time applications such as RAG-based chatbots. Its optimized hardware architecture enables low-latency responses.

### Difference Between Groq RAG and OpenAI RAG

Groq focuses on high-speed inference using specialized hardware, whereas OpenAI models are optimized for general-purpose reasoning tasks.

### Role of Streamlit in Rapid GenAI Prototyping

Streamlit allows developers to quickly build interactive user interfaces for AI applications with minimal code, making it ideal for prototyping GenAI systems.

---

## Conclusion

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be implemented to build a document-based question-answering system.

By combining **LangChain, FAISS, HuggingFace embeddings, and Groq LLMs**, the system retrieves relevant information from uploaded documents and generates accurate responses grounded in that context.

Streamlit enables rapid development of the chatbot interface, making it easy for users to interact with their documents conversationally.

The project highlights how RAG systems improve factual accuracy and reduce hallucinations in modern AI applications.


# GitHub Repository

https://github.com/sdas45git/chatgroq-rag-streamlit


---
