import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables
load_dotenv()

# LangChain prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You are a helpful assistant. Please respond to the question asked."},
        {"role": "user", "content": "question: {question}"}
    ]
)

# Streamlit interface
st.title(":chains: Langchain Demo with Gemma 2B")

input_text = st.text_input("What question do you have in mind?")

# Call Ollama model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()

# Combine prompt, model, and parser into a chain
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
