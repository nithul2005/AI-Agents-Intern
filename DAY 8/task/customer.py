import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")

file_path = input("📄 Enter path to your PDF file: ").strip()
loader = PyPDFLoader(file_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-70b-8192",
    temperature=0.5
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and professional AI assistant for answering customer FAQs.
Use only the information provided below to answer the user's question.

[Context]
{context}

[User Question]
{question}

[Helpful Answer]
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

print("\n🤖 PDF FAQ Chatbot is ready. Ask your questions!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = qa_chain.run(user_input)
    print("Bot:", response, "\n")
