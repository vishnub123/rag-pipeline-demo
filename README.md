# Retrieval-Augmented Generation (RAG) Pipeline with ChromaDB, HuggingFace, and Groq Llama3

This project implements a Retrieval-Augmented Generation (RAG) pipeline for intelligent question answering over custom document collections. It combines fast document retrieval using ChromaDB and HuggingFace embeddings with advanced language generation via Groq’s Llama3 model, orchestrated through LangChain.

## Features

- **Document Ingestion:** Load and process text documents from a directory.
- **Text Chunking:** Split documents into manageable chunks for efficient retrieval.
- **Embeddings Generation:** Use HuggingFace’s `all-MiniLM-L6-v2` model to generate vector embeddings for each chunk.
- **Vector Database:** Store and search document chunks using ChromaDB for fast, semantic retrieval.
- **Contextual Question Answering:** Retrieve relevant chunks based on user queries and generate context-aware answers using Groq’s Llama3 model.
- **Modular Pipeline:** Easily extend or adapt for other document types, embedding models, or LLMs.

## How It Works

1. **Load Documents:** Text files are loaded from a specified directory.
2. **Chunking:** Each document is split into overlapping chunks for granular retrieval.
3. **Embedding:** Chunks are converted into vector embeddings using HuggingFace’s transformer model.
4. **Storage:** Embeddings and chunk metadata are stored in a persistent ChromaDB collection.
5. **Querying:** User questions are embedded and used to retrieve the most relevant chunks from the database.
6. **Generation:** Retrieved context and the question are sent to Groq’s Llama3 model via LangChain, producing a context-aware answer.

## Usage

1. **Install dependencies:**  
   Install required Python packages using your preferred package manager.

2. **Configure environment:**  
   - Place your Groq API key in a `.env` file as `GROQ_API_KEY=your_key_here`.
   - Add your text documents (e.g., news articles) to the `news_articles` directory.

3. **Run the pipeline:**  
   Execute `main.py` to load documents, generate embeddings, store them in ChromaDB, and interact with the RAG system.

## Example

```python
question = "Tell me about AI replacing TV writers strike"
relevant_chunks = query_documents(question, n_results=5)
response = generate_response(question, relevant_chunks)
print("LLM Response:", response)
```

## Technologies Used

- Python
- ChromaDB (vector database)
- HuggingFace Sentence Transformers
- LangChain
- Groq Llama3 API

## Getting Started

See the code for detailed implementation and customization options.  
Make sure to exclude `.env`, `.venv`, and `chroma_persistent_storage/` from version control (see `.gitignore`).

---

**This project is ideal for building intelligent, context-aware QA systems over your own document collections using state-of-the-art retrieval and generation**
