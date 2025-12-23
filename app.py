import streamlit as st
from rag_backend import generate_rag_answer, retrieve_docs, learn_new_information

st.title("Team X RAG System")

query = st.text_input("Enter your query")

# Initialize session state
if "thumbs_down" not in st.session_state:
    st.session_state.thumbs_down = False
if "search_done" not in st.session_state:
    st.session_state.search_done = False
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if st.button("Search"):
    if query:
        st.session_state.retrieved_docs = retrieve_docs(query)
        st.session_state.answer = generate_rag_answer(query)
        st.session_state.search_done = True
        st.session_state.thumbs_down = False  # Reset on new search
    else:
        st.error("Please enter a query.")

# Display results if search was done
if st.session_state.search_done:
    st.subheader("Answer:")
    st.write(st.session_state.answer)
    
    # Display Sources
    st.subheader("Sources Used:")
    if st.session_state.retrieved_docs:
        for i, doc in enumerate(st.session_state.retrieved_docs, 1):
            st.write(f"{i}. {doc[:100]}...")  # Truncate for display
    else:
        st.write("No sources found.")
    
    # Feedback section
    st.subheader("Was this helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Thumbs Up"):
            st.success("Thanks for the feedback!")
            st.session_state.thumbs_down = False
    with col2:
        if st.button("👎 Thumbs Down"):
            st.session_state.thumbs_down = True
    
    # Show correction input if thumbs down
    if st.session_state.thumbs_down:
        st.warning("Sorry to hear that. Please provide the correct answer below.")
        correct_answer = st.text_input("What is the correct answer?")
        if st.button("Submit Correction"):
            if correct_answer:
                message = learn_new_information(correct_answer)
                st.success(message)
                st.session_state.thumbs_down = False
            else:
                st.error("Please enter a correct answer.")