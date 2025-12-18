import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()


# Llama3-8b is fast and excellent for RAG
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def test_connection():
    print("🤖 Testing connection to LLM...")
    try:
        # Simple test prompt
        response = llm.invoke("Explain what 'RAG' (Retrieval Augmented Generation) is in one sentence.")
        
        print("\n✅ Connection Successful!")
        print(f"📝 Model Answer:\n{response.content}")
        return True
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()