import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MADS Program Chat Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl"

@st.cache_resource
def load_or_create_embeddings():
    """Load embeddings from cache or create new ones"""
    try:
        if EMBEDDINGS_CACHE.exists():
            with EMBEDDINGS_CACHE.open("rb") as f:
                splits = pickle.load(f)
            st.session_state.embeddings_loaded = True
            return splits
        else:
            # Show loading message
            with st.spinner("First-time setup: Loading documents (this may take a few minutes)..."):
                # Load and process documents
                with open('unique_links_list.txt', 'r') as file:
                    urls = [line.strip() for line in file.readlines()]
                
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=200,
                    add_start_index=True
                )
                splits = text_splitter.split_documents(data)
                
                # Cache the results
                with EMBEDDINGS_CACHE.open("wb") as f:
                    pickle.dump(splits, f)
                
                st.session_state.embeddings_loaded = True
                return splits
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

@st.cache_resource
def initialize_rag_chain(_api_key):
    """Initialize the RAG chain with necessary components"""
    try:
        # Set the API key
        os.environ['OPENAI_API_KEY'] = _api_key
        
        # Load or create embeddings
        splits = load_or_create_embeddings()
        if splits is None:
            return None
            
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()
        
        # Create prompt template
        template = """Please provide a concise answer based on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Using GPT-3.5 for faster responses
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
    except Exception as e:
        st.error(f"Error initializing the chat assistant: {str(e)}")
        return None

# Main app interface
st.title("🎓 MADS Program Chat Assistant")

# Show loading progress
if not st.session_state.embeddings_loaded:
    st.info("⚙️ Setting up the chat assistant for first use... This will only happen once.")

# API Key input section
if not st.session_state.api_key:
    with st.container():
        api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if st.button("Submit API Key"):
            if api_key.startswith('sk-'):
                st.session_state.api_key = api_key
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API key")
else:
    # Initialize RAG chain if needed
    if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
        st.session_state.rag_chain = initialize_rag_chain(st.session_state.api_key)

    # Chat interface
    for message in st.session_state.messages:
        role = "assistant" if message["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the MADS program"):
        with st.chat_message("user"):
            st.write(question)
        
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(question)
                st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.rag_chain = None
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by LangChain and OpenAI")
