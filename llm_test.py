import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def test_connection():
    
    try:
        # Simple test prompt
        response = llm.invoke("Explain what RAG is in one sentence.")
        
        print("\nConnection Successful")
        print(f"Model Answer:\n{response.content}")
        return True
    except Exception as e:
        print(f"\nConnection Failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()