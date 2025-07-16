import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

from dotenv import load_dotenv
load_dotenv()

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Write an engaging short story based on the following input: {query}. "
        "Blend multiple genres such as mystery, fantasy, adventure, and romance where appropriate. "
        "Include vivid characters, surprising plot twists, and emotional depth. "
        "Keep it imaginative, concise, and suitable for general readers."
    )
)


# Initialize Groq LLM
llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit App UI
st.set_page_config(page_title="LangChain + Groq", layout="centered")
st.title("🧠 LangChain + Groq with LLaMA3")
st.markdown("Ask your query below and get an AI-powered response.")

# Input box for user query
user_query = st.text_input("Enter your query here:")

# Display output when button clicked
if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            response = chain.run(user_query)
        st.success("Response:")
        st.write(response)
