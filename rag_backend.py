import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. Load Environment & Constants
load_dotenv()
EMBEDDINGS_DIR = "embeddings" 
INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "wiki_faiss.index")
DOC_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "doc_store.json")

# 2. Setup LLM (The Brain)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 3. Setup Embedding Model 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Load the Vector Database (The Memory)
try:
    index = faiss.read_index(INDEX_FILE)
    with open(DOC_STORE_FILE, "r", encoding="utf-8") as f:
        doc_store = json.load(f)
    print("Database loaded successfully")
except Exception as e:
    print(f"Error loading database: {e}")
    index = None
    doc_store = {}

# --- CORE LOGIC ---

def retrieve_docs(query, top_k=3):
    if index is None:
        return ["Error: Database not loaded."]

    # 1. Convert query to vector
    query_vector = embedding_model.encode([query]).astype("float32")
    
    # 2. Normalize 
    faiss.normalize_L2(query_vector)
    
    # 3. Search FAISS
    distances, indices = index.search(query_vector, top_k)
    
    # 4. Fetch actual text from doc_store
    results = []
    for idx in indices[0]:
        # returns -1 if it finds nothing
        if idx != -1:
            # convert numpy int to string key for JSON lookup
            doc_id = str(idx) 
            if doc_id in doc_store:
                results.append(doc_store[doc_id]["text"])
    
    return results

def generate_rag_answer(query):
    """
    Full RAG Pipeline: Retrieval + Generation
    """
    # Step 1: Retrieve context
    retrieved_chunks = retrieve_docs(query)
    
    if not retrieved_chunks:
        return "I couldn't find any relevant information in the database."
    

    # Step 2: System Prompt
    template = """
    You are a helpful AI assistant for a university project.
    
    INSTRUCTIONS:
    1. Answer the user's question based ONLY on the context provided below.
    2. If the answer is not in the context, say "I don't have enough information."
    3. Citation: If you use a specific fact, try to mention the source if available.
    
    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Join chunks for the LLM
    context_text = "\n\n".join(retrieved_chunks)
    
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    return response


def learn_new_information(user_correction):
    """
    self learning: Adds user corrections to the database dynamically.
    """
    global index, doc_store
    
    print(f"Learning: {user_correction[:30]}...")
    
    # 1. Create new ID (simple increment)
    new_id = str(len(doc_store))
    
    # 2. Embed the text
    new_embedding = embedding_model.encode([user_correction]).astype("float32")
    faiss.normalize_L2(new_embedding)
    
    # 3. Add to FAISS Index
    index.add(new_embedding)
    
    # 4. Add to Doc Store
    doc_store[new_id] = {"text": user_correction, "source": "User Feedback"}
    
    # 5. Save updates to disk 
    faiss.write_index(index, INDEX_FILE)
    with open(DOC_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_store, f, indent=4)
        
    return "I have learned this new information"


# --- TEST FUNCTION ---
if __name__ == "__main__":
    # Test with a query relevant to dataset
    user_query = "How fast can modern computers calculate?"
    
    print("-" * 30)
    print(f"Query: {user_query}")
    print("-" * 30)
    
    answer = generate_rag_answer(user_query)
    
    print(f"Generated Answer:\n{answer}")
    print("-" * 30)