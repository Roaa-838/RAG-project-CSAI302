import streamlit as st
from rag_backend import generate_rag_answer, retrieve_docs, learn_new_information

st.title("Team X RAG System")

query = st.text_input("Enter your query")


if st.button("Search"):
    if query:
        # Retrieve docs for sources
        retrieved_docs = retrieve_docs(query)
        answer = generate_rag_answer(query)
        st.subheader("Answer:")
        st.write(answer)
        
        # Display Sources
        st.subheader("Sources Used:")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"{i}. {doc[:100]}...")  # Truncate for display
        else:
            st.write("No sources found.")
        
        # Feedback section
        st.subheader("Was this helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Thumbs Up"):
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎 Thumbs Down"):
                st.warning("Sorry to hear that. Please provide the correct answer below.")
                correct_answer = st.text_input("What is the correct answer?")
                if st.button("Submit Correction"):
                    if correct_answer:
                        message = learn_new_information(correct_answer)
                        st.success(message)
                    else:
                        st.error("Please enter a correct answer.")
    else:
        st.error("Please enter a query.")