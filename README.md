# 📚 AI Study Assistant (RAG-based PDF Q&A)

🔗 **Live App:**  
https://aditya-raj4-rag-based-ai-study-assistant-app-bwuj0a.streamlit.app/
https://rag-based-ai-study-assistant.onrender.com

---

## 🚀 Overview

**AI Study Assistant** is a Retrieval-Augmented Generation (RAG) based web application that allows users to upload PDF documents and interact with them using natural language queries.

Users can ask questions, generate summaries, and create quiz questions directly from the uploaded PDF.  
This project demonstrates the practical application of **LLMs, vector databases, and LangChain** without training any custom machine learning models.

---

## ✨ Features

- 📄 Upload PDF documents  
- 🔍 Ask questions based on PDF content  
- 📝 Generate short summaries  
- 🧪 Create quiz questions automatically  
- ⚡ Fast semantic search using FAISS  
- 🌐 Deployed on Streamlit Cloud  

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF Loading** – Extracts text from the uploaded PDF  
2. **Text Chunking** – Splits content into manageable chunks  
3. **Embeddings** – Converts text into vector embeddings  
4. **Vector Store (FAISS)** – Retrieves relevant chunks efficiently  
5. **LLM (FLAN-T5)** – Generates responses using retrieved context  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Google FLAN-T5 (HuggingFace)  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector Database:** FAISS  
- **Framework:** LangChain  
- **Language:** Python  
