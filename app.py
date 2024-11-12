import streamlit as st
from langchain_openai import ChatOpenAI
import pickle
import os
import numpy as np
from typing import List

# Page config
st.set_page_config(
    page_title="MADS Program Chat Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

@st.cache_resource
def load_preprocessed_data():
    """Load the preprocessed data from pickle file"""
    try:
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessed data: {str(e)}")
        return None

def find_relevant_texts(query: str, preprocessed_data: dict, k: int = 3) -> List[str]:
    """Find the most relevant texts using cosine similarity"""
    from langchain_openai import OpenAIEmbeddings
    
    # Get query embedding
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities
    similarities = [
        np.dot(query_embedding, doc_embedding) / 
        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        for doc_embedding in preprocessed_data['embeddings']
    ]
    
    # Get top k most similar texts
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [preprocessed_data['texts'][i] for i in top_k_indices]

# Main app interface
st.title("🎓 MADS Program Chat Assistant")

# Load preprocessed data
if st.session_state.preprocessed_data is None:
    with st.spinner("Loading knowledge base..."):
        st.session_state.preprocessed_data = load_preprocessed_data()
        if st.session_state.preprocessed_data is None:
            st.error("Failed to load knowledge base")
            st.stop()

# API Key input
if not st.session_state.api_key:
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API Key:", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key.startswith('sk-'):
            st.session_state.api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            st.rerun()
        elif submitted:
            st.error("Please enter a valid OpenAI API key")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask about the MADS program"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get relevant context
                relevant_texts = find_relevant_texts(
                    question, 
                    st.session_state.preprocessed_data
                )
                context = "\n\n".join(relevant_texts)
                
                # Generate response using ChatGPT
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0
                )
                
                response = chat.predict(
                    f"""Based on this context: {context}
                    
                    Answer this question: {question}
                    
                    Give a clear and concise answer."""
                )
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by LangChain and OpenAI")
