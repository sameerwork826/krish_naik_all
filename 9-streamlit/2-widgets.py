import streamlit as st 


st.title("Streamlit text input")

name=st.text_input("Enter you name")

age=st.slider("select your age:" ,0,100,25)

options=['python','jaa','c++','ruby']
choice=st.selectbox("choose a language",options)
st.write(f"you selected {choice}")
if name:
    st.write(f"hello ,{name}")
    st.write(f"your age is {age}")
    
    