import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.title("RAG Chatbot")

api_key = st.text_input("Enter OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.read())

        loader = TextLoader("temp.txt")
        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

        retriever = db.as_retriever()

        llm = ChatOpenAI(model="gpt-3.5-turbo")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        question = st.text_input("Ask a question")

        if question:
            answer = qa_chain.run(question)
            st.write("### Answer")
            st.write(answer)
