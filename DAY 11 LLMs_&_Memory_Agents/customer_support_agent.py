import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_knowledge_base(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename), encoding="utf-8")
            docs.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

folder_path = "knowledge_base"
raw_docs = load_knowledge_base(folder_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def chat():
    print("🤖 Customer Support Agent Ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break
        answer = qa_chain.run(query)
        print(f"Agent: {answer}\n")

if __name__ == "__main__":
    chat()
