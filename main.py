import os
import streamlit as st
from dotenv import load_dotenv
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

# 2. Initialize and Cache the RAG Pipeline
@st.cache_resource
def build_rag_pipeline():
    # Note: Using the exact path relative to main.py
    loader = TextLoader("data/genai_course.txt", encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    # Retrieve documents and format them into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(model_name="openai/gpt-oss-20b")
    
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
rag_chain = build_rag_pipeline()

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