import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load API keys from .env
load_dotenv()

# 1. Page Configuration
st.set_page_config(page_title="Course Q&A Assistant", page_icon="🤖")
st.title("📚GenAI Course Assistant")
st.write("Ask anything about the Master Generative AI & System Architecture course.")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        groq_api_key = None

if not groq_api_key:
    st.error("Set GROQ_API_KEY in the deployment's environment variables.")
    st.stop()

# 2. Initialize and Cache the RAG Pipeline
@st.cache_resource
def build_rag_pipeline(api_key: str):
    course_file = Path(__file__).parent / "data" / "genai_course.txt"
    loader = TextLoader(str(course_file), encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    # Retrieve documents and format them into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=api_key)
    
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    # Build the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Load the pipeline into memory once
rag_chain = build_rag_pipeline(groq_api_key)

# 3. User Interface for Querying
user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Searching course materials..."):
        try:
            # Pass the user's question directly to the chain
            response = rag_chain.invoke(user_question)
            st.markdown("### Answer")
            st.info(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")