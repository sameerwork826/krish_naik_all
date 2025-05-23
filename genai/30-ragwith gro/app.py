import streamlit as st  # Fixed typo in import
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings  # Fixed class name
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_docs_chain
from langchain.chains import create_retrieval_chain  # Fixed typo
from langchain_community.vector_stores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Fixed typo

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")  # Removed duplicate line

llm = ChatGroq(model="gemma-2b-it", groq_api_key=groq_api_key)  # Fixed model name

prompt = ChatPromptTemplate.from_template(  # Fixed prompt creation
    """Answer the question based on the context provided. If the question cannot be answered based on the context, say "I don't know".
    
    Context: {context}
    Question: {input}"""
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()  # Fixed class name
        st.session_state.loader = PyPDFDirectoryLoader("Research_Papers")  # Fixed directory name typo
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.texts = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.texts, st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from the paper")  # Fixed variable name

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Embedding Created")

import time

if user_prompt:
    document_chain = create_stuff_docs_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(  # Fixed typo and removed incorrect method call
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever()
    )
    
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})  # Fixed variable name typo
    end = time.process_time()
    st.write("Time taken to process the query: ", end-start)
    
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"Document {i+1}: {doc.page_content}")
        st.write("Answer: ", response["answer"])  # Fixed key and removed stray 'k'