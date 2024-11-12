import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

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
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

@st.cache_resource
def load_vectorstore(api_key):
    """Load the preprocessed vector store"""
    try:
        os.environ['OPENAI_API_KEY'] = api_key
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("processed_data.faiss", embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

@st.cache_resource
def initialize_rag_chain(vectorstore):
    """Initialize the RAG chain with the loaded vector store"""
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Your custom prompt template
    template = """Please provide enough detail in your answer. If applicable, you may provide an URL relavent to the question asked. Answer the question based on the following context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create and return RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main app interface
st.title("🎓 MADS Program Chat Assistant")

# Check for vectorstore
if not os.path.exists("processed_data.faiss"):
    st.error("""
    Preprocessed data not found! Please run preprocess.py first.
    Check the README.md for instructions.
    """)
    st.stop()

# API Key input
if not st.session_state.api_key:
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API Key:", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key.startswith('sk-'):
            st.session_state.api_key = api_key
            st.rerun()
        elif submitted:
            st.error("Please enter a valid OpenAI API key")

else:
    # Load vectorstore and initialize RAG chain if needed
    if st.session_state.rag_chain is None:
        with st.spinner("Loading knowledge base..."):
            vectorstore = load_vectorstore(st.session_state.api_key)
            st.session_state.rag_chain = initialize_rag_chain(vectorstore)

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
                response = st.session_state.rag_chain.invoke(question)
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

        # Add some helpful information
        st.markdown("---")
        st.markdown("""
        ### About this Assistant
        This chatbot helps you learn about the UChicago MADS program by:
        - Providing accurate program information
        - Answering admission questions
        - Explaining course details
        - Sharing relevant URLs
        """)

# Footer
st.markdown("---")
st.caption("Powered by LangChain and OpenAI")
