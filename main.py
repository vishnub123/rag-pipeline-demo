import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a wrapper to use with ChromaDB
class HuggingFaceEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        pass


    def __call__(self, input):
        return embedding_model.encode(input).tolist()
    
    def name(self):
        return "huggingface"

# Initialize chromadb client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="rag_collection_hf",
    embedding_function=HuggingFaceEmbeddingFunction()
)

# Groq Chat client setup
llm = ChatGroq(
    model_name="llama3-70b-8192", 
    api_key=groq_api_key
)

# Example Prompt
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that can answer questions based on the provided context.\n\nQuestion: {question}"
)

chain = prompt | llm

# Call the model
response = chain.invoke({"question": "What is Retrieval-Augmented Generation (RAG)?"})
# print("LLM Response:", response)


#function to load documents from a directory
def load_documents_from_directory(directory):
    print("====Loading documents from directory====")
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({"id": filename, "text": content})
    return documents

#function to split text into chunks
def text_split(text,chunk_size=1000,chunk_overlap=20):
    chunks=[]
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

#load documents from a directory
directory_path = "./news_articles"  
document = load_documents_from_directory(directory_path)
print(f"Loaded {len(document)} documents from {directory_path}")


#split documents into chunks
chunked_documents = []
for doc in document:
    chunks = text_split(doc['text'])
    print("===Splitting docs into chunks===")
    for chunk in chunks:
        chunked_documents.append({"id": doc['id'], "text": chunk})
print(f"Split documents into {len(chunked_documents)} chunks.")


# Function to generate embeddings using HuggingFace model
def get_hf_embeddings(text):
    return embedding_model.encode(text).tolist()

#generate embeddings for document chunks
for doc in chunked_documents:
    print("Generating embeddings....")
    doc['embedding'] = get_hf_embeddings(doc['text'])

print(doc["embedding"])

#function to query documents
def query_documents(query, n_results=5):
    print("====Querying documents====")
    query_embedding = get_hf_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    #extract relevant chunks
    relevant_chunks = [doc for sublist in results['documents'] for doc in sublist]
    print("===Returning relevant chunks===")
    return relevant_chunks



#function to generate response from LLM
def generate_response(question, relevant_chunks):
    print("====Generating response from LLM====")
    context = "\n".join([chunk for chunk in relevant_chunks])
    prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that can answer questions based on the provided context.\n\nContext: {context}\n\nQuestion: {question}"
   )
    # Use the chain (prompt | llm) for invocation
    response = chain.invoke({"question": question, "context": context})
    return response.content if hasattr(response, "content") else response

# Example RAG pipeline usage
question = "Tell me about AI replacing TV writers strike"
relevant_chunks = query_documents(question, n_results=5)
response = generate_response(question, relevant_chunks)
print("LLM Response:", response)

    




