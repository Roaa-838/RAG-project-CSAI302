import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load keys
load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# --- YOUR CORE LOGIC ---

def mock_retrieve(query):
    """
    TEMPORARY FUNCTION: Simulates Mariam's Vector DB.
    Use this to test your prompt engineering without waiting for the DB.
    """
    print(f"DEBUG: Mock searching for '{query}'...")
    
    # Simulating that we found these 2 documents in the database
    return [
        "Document A: RAG systems combine a retriever and a generator.",
        "Document B: The retrieval step finds relevant docs, and the generator creates the answer."
    ]

def generate_rag_answer(query, context_chunks):
    """
    Task 3.2: The System Prompt.
    This forces the model to use ONLY the provided context.
    """
    
    # 1. Define the Strict System Prompt
    template = """
    You are a helpful AI assistant for a university project.
    
    INSTRUCTIONS:
    1. Answer the user's question based ONLY on the context provided below.
    2. If the answer is not in the context, say "I don't have enough information in my database."
    3. Do not make up facts.
    
    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 2. Create the Chain (Prompt -> LLM -> String Output)
    chain = prompt | llm | StrOutputParser()
    
    # 3. Invoke the chain
    # Join the list of chunks into one big string
    context_text = "\n\n".join(context_chunks)
    
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    return response

# --- TEST FUNCTION (Run this file directly to test) ---
if __name__ == "__main__":
    user_query = "How does RAG work?"
    
    # Step 1: Get (Mock) Data
    # Later, you will replace this line with: retrieved_docs = mariam_search_function(user_query)
    retrieved_docs = mock_retrieve(user_query) 
    
    # Step 2: Generate Answer
    final_answer = generate_rag_answer(user_query, retrieved_docs)
    
    print("-" * 30)
    print(f"❓ Query: {user_query}")
    print("-" * 30)
    print(f"💡 Generated Answer:\n{final_answer}")
    print("-" * 30)