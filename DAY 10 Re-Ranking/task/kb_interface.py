import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
import gradio as gr

os.environ["GROQ_API_KEY"] = ""

def load_documents():
    docs = []
    for file in os.listdir("data"):
        path = os.path.join("data", file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def build_knowledge_base():
    raw_docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

db = build_knowledge_base()
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(model_name="llama3-8b-8192")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked]

def chat_interface(message, history):
    docs = retriever.get_relevant_documents(message)
    reranked_docs = rerank(message, docs)
    sub_db = FAISS.from_documents(reranked_docs, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    sub_retriever = sub_db.as_retriever()
    sub_chain = RetrievalQA.from_chain_type(llm=llm, retriever=sub_retriever)
    response = sub_chain.run(message)
    return response

ui = gr.ChatInterface(fn=chat_interface, title="📚 Personal Knowledge Base Assistant")

ui.launch()
