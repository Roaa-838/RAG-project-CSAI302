# 🧠 AI-Powered RAG System (CSAI 302 Project)

> **Team Members:**
> * **Mariam Alhaj:** Data & Infrastructure
> * **Roaa Raafat:** Core Logic & RAG Engineering
> * **Yousef ElDawayaty:** Frontend & Self-Learning System

## 📖 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer user queries based on a specific knowledge base (Simple English Wikipedia). It combines a **Vector Database (FAISS)** for semantic search with a **Large Language Model (Llama 3.1-8b)** to generate accurate, context-aware responses.

The system features a **Self-Learning Layer** that allows it to improve over time by incorporating user feedback and corrections directly into its memory.

---

## 🏗️ System Architecture

**Data Flow:**
`PDF/Text Data` $\to$ `Preprocessing & Chunking` $\to$ `Embedding Model (all-MiniLM-L6-v2)` $\to$ `Vector DB (FAISS)` $\to$ `Retrieval` $\to$ `LLM Generation` $\to$ `Streamlit UI`

**Key Components:**
* **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers) - Generates 384-dimensional dense vectors.
* **Vector Database:** `FAISS` (Facebook AI Similarity Search) using `IndexFlatIP` for Cosine Similarity.
* **LLM:** `Llama-3.1-8b-instant` (via Groq API) for fast, grounded generation.
* **Interface:** `Streamlit` for a friendly, interactive web UI.

---

## ✨ Features

### 1. **Semantic Search & Retrieval**
* Uses **Cosine Similarity** to find the most relevant document chunks, even if the user doesn't use exact keywords.
* Retrieves top-3 relevant contexts to ground the LLM's answer.

### 2. **Context-Aware Generation**
* Strict "Retrieve-then-Generate" pipeline prevents hallucinations.
* The LLM refuses to answer if the information is not present in the database.

### 3. **Self-Learning Mechanism (Bonus)**
* **Feedback Loop:** Users can vote "Thumbs Down" on an answer.
* **Real-time Update:** Users provide the *correct* answer, which is immediately embedded and added to the FAISS index.
* **Memory:** The system "learns" this new fact and will answer correctly the next time the question is asked.

---

## 🚀 Installation & Setup

### **Prerequisites**
* Python 3.8+
* A Groq API Key (Free tier available)

### **Step 1: Clone the Repository**
```bash
git clone [https://github.com/roaa-838/RAG-project-CSAI302.git](https://github.com/roaa-838/RAG-project-CSAI302.git)
cd RAG-project-CSAI302
```

### **Step 2: Install Dependencies**
```bash
# Create a virtual environment (Recommended)
python -m venv venv
# Activate it (Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate)

# Install libraries
pip install -r requirements.txt
```
### **Step 3: Configure API Keys**
* Create a .env file in the root directory and add your Groq API key:
```bash
GROQ_API_KEY=gsk_your_actual_api_key_here
```
### **Step 4: Initialize the Database (One-time Setup) **
Note: The repo may already contain pre-built indices in embeddings/. If not, run:
```bash
# Processes the data in 'data/' and builds the FAISS index
python src/ingest.py
```

### **Step 5: Run the Application**
```bash
streamlit run src/app.py
```
The application will open in your browser at http://localhost:8501

## Project Structure
```text
RAG-Project/
├── 📂 data/               # Raw PDF/Text files
├── 📂 embeddings/         # FAISS index and metadata (The "Memory")
├── 📂 src/
│   ├── app.py             # Streamlit UI (Frontend)
│   ├── ingest.py          # Database creation script 
│   └── rag_backend.py     # Core RAG Logic & Self-Learning 
├── .env                   # API Keys (Not uploaded to Git)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🧪 Testing & Demonstration
Sample Query 1: "How fast can modern computers calculate?"

* Result: Retrieves facts about Pi calculation speed (31.4 trillion digits).
* Status: ✅ Verified

Sample Query 2: "How much faster is light compared to the Earth?"

* Result: Retrieves the specific comparison factor (10,210 times faster).
* Status: ✅ Verified
