"""
RAG Pipeline

- Prepare documents and ingest them into a Chroma database for retrieval. 
- Create a prompt template to include the retrieved chunks from the source 
  documents and answer user questions.
- Use the rag_chain to answer questions about the source documents.
"""
import os
import tempfile
import requests

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# TODO: Refactor code to avoid loading, storing, and embedding the same documents multiple times. 
# Instead, load the documents once, store them in a vector database, and then retrieve them 
# when needed.
# TODO: Determine the best chunk size and overlap for the text splitter.

pdf_path = "https://arxiv.org/pdf/2401.08406"  # rag_vs_fine_tuning.pdf
csv_path = "https://raw.githubusercontent.com/fivethirtyeight/data/refs/heads/master/fifa/fifa_countries_audience.csv"
html_path = "https://bidenwhitehouse.archives.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/"

CHUNK_RECURSIVE_SEPARATORS = ["\n", " ", ""]
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3.2"
VECTORSTORE_PERSIST_DIRECTORY = "./chroma_db"
VECTORSTORE_SEARCH_TYPE = "similarity"
VECTORSTORE_SEARCH_K = 3

# ------------------------------------------------------------
# Load PDF document
# PyPDFLoader expects a path to a local or remote PDF file and returns a list of Document objects
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

print("Total PDF documents loaded:", len(pdf_docs))

# ------------------------------------------------------------
# Load CSV document
# CSVLoader expects a path to a local CSV file and returns a list of Document objects.
# First, download the CSV file because CSVLoader doesn't support remote URLs
response = requests.get(csv_path, timeout=30)
response.raise_for_status()
csv_bytes = response.content

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(csv_bytes)
    tmp_path = tmp.name

csv_loader = CSVLoader(tmp_path)
csv_docs = csv_loader.load()

# Remove temporary file if it exists
if os.path.exists(tmp_path):
    os.remove(tmp_path)

print("Total CSV documents loaded:", len(csv_docs))

# ------------------------------------------------------------
# Load HTML document
# UnstructuredURLLoader expects a list of URLs and returns a list of Document objects
html_loader = UnstructuredURLLoader(urls=[html_path])
html_docs = html_loader.load()

print("Total HTML documents loaded:", len(html_docs))

# ------------------------------------------------------------
# Split source documents into chunks for embedding
source_documents = pdf_docs + csv_docs + html_docs

text_splitter = RecursiveCharacterTextSplitter(
    separators=CHUNK_RECURSIVE_SEPARATORS,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP)

document_chunks = text_splitter.split_documents(source_documents)

print(f"Total source documents: {len(source_documents)}")
print(f"Total document chunks: {len(document_chunks)}")
print(f"First chunk preview:\n{document_chunks[0].page_content}")

# ------------------------------------------------------------
# Embed document chunks in a persistent Chroma vector database
# Using Ollama for embeddings (no API key needed)
# Make sure you have pulled the embedding model: ollama pull nomic-embed-text
embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# The .from_documents() method to embed and ingest the documents into a Chroma vector
# database with the provided embeddings function.
vectorstore = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_function,
    persist_directory=VECTORSTORE_PERSIST_DIRECTORY
)

print(f"Document chunks embedded and persisted to {VECTORSTORE_PERSIST_DIRECTORY}")
print(f"Total chunks stored in vector database: {vectorstore._collection.count()}")

# Configure the vector store as a retriever object that returns the top 3 documents 
# for use in the final RAG chain
retriever = vectorstore.as_retriever(
    search_type=VECTORSTORE_SEARCH_TYPE,
    search_kwargs={"k": VECTORSTORE_SEARCH_K}
)

# ----------------------------------------------------------

# We need to design a chat prompt template to combine the retrieved document chunks 
# with the user input question

message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Initialize Ollama LLM for generating responses
# Make sure you have pulled the model: ollama pull llama3.2
llm = ChatOllama(model=OLLAMA_LLM_MODEL)

# Create a chain to link retriever, prompt_template, and llm.
# The retriever receives the question string directly and retrieves relevant documents.
# RunnablePassthrough passes the question to the prompt template.
rag_chain: Runnable = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
)

# Invoke the chain with the question as a string
question = "What is the case study on agriculture in the paper about?"
response = rag_chain.invoke(question)
print(response.content)
