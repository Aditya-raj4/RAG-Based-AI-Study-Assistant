import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

st.set_page_config(page_title="AI Study Assistant")
st.title("📚 AI Study Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # ✅ Hugging Face embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # ✅ Hugging Face LLM (FREE, local)
    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    st.success("PDF indexed successfully ✅")

    query = st.text_input("Ask a question from the PDF")

    if query:
        st.subheader("Answer")
        st.write(qa.run(query))

        if st.button("Summarize"):
            st.write(qa.run("Give a short summary of this document"))

        if st.button("Quiz"):
            st.write(qa.run("Create 3 quiz questions from this document"))

