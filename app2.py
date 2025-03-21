import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Google API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

#os.environ["GOOGLE_API_KEY"] = "AIzaSyAlBYH03S1F583dVX-5ae8nyTmvIycdd0o"
st.title("RAG Application built on Gemini Model")

# Load PDF
loader = PyPDFLoader(r"PDFS\yolov9_paper.pdf")

# Split document into pages (e.g., pages in groups of 10)
page_size = 10  # Group pages in chunks of 10 pages
num_pages = len(loader.load())  # Get the total number of pages
chunks = []

for i in range(0, num_pages, page_size):
    pages = loader.load()[i:i + page_size]
    chunk = "".join([page.page_content for page in pages])  # Combine text of multiple pages into one chunk
    chunks.append(chunk)

# Split text into chunks with a smaller size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Create list of Document objects
docs = []
for chunk in chunks:
    doc = Document(page_content=chunk)  # Wrap chunk as Document
    docs.extend(text_splitter.split_documents([doc]))  # Split and add the result to docs list

# Create vector store with embeddings and persist directory
# Create vector store with embeddings and persist directory
persist_directory = "chroma_db"  # Choose a directory for persistence
if not os.path.exists(persist_directory):  # only create the vector store if it does not already exist.
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=persist_directory,
    )
    vectorstore.persist()  # persist immediately after creation.
else:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Chat input
query = st.chat_input("Say something:")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Process query if given
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])