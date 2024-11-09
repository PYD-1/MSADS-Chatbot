import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Page configuration
st.set_page_config(
    page_title="MADS Program Chat Assistant",
    page_icon="🎓",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .bot-message {
        background-color: #f0f2f6;
    }
    .api-key-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

def load_urls():
    with open('unique_links_list.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

@st.cache_resource
def initialize_rag_chain(api_key):
    """Initialize the RAG chain with necessary components"""
    try:
        # Set the API key
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Load URLs
        urls = load_urls()
        
        # Load documents
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(data)
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()
        
        # Create prompt template
        template = """Please provide answer based on the following context:
        {context}. You may also search online if you can't find the answer in context.

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create LLM
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
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

# Main app header
st.title("🎓 MADS Program Chat Assistant")

# API Key input section
if not st.session_state.api_key:
    st.markdown("""
    ### Welcome to the MADS Program Chat Assistant!
    To get started, please enter your OpenAI API key below.
    
    > Don't have an API key? Get one from [OpenAI's website](https://platform.openai.com/api-keys)
    """)
    
    with st.container():
        st.markdown('<div class="api-key-container">', unsafe_allow_html=True)
        api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if st.button("Submit API Key"):
            if api_key.startswith('sk-') and len(api_key) > 40:
                st.session_state.api_key = api_key
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API key. It should start with 'sk-'")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some helpful information
        with st.expander("ℹ️ About API Keys and Safety"):
            st.markdown("""
            - Your API key is used only for this session and is not stored permanently
            - The key is stored in your browser's session storage
            - The session ends when you close your browser
            - Each API call will use your OpenAI credits
            - For safety, never share your API key with others
            """)
else:
    try:
        # Initialize RAG chain if not already done
        if st.session_state.rag_chain is None:
            with st.spinner("Initializing the chat assistant... This may take a minute."):
                st.session_state.rag_chain = initialize_rag_chain(st.session_state.api_key)

        # Add option to reset API key
        if st.sidebar.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.rag_chain = None
            st.session_state.messages = []
            st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>You:</b><br>{message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message bot-message">
                            <b>Assistant:</b><br>{message["content"]}
                        </div>
                    """, unsafe_allow_html=True)

        # Chat input
        if question := st.chat_input("Ask a question about the MADS program"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(question)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.rerun()

        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Reset Application"):
            st.session_state.clear()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by LangChain and OpenAI</p>
</div>
""", unsafe_allow_html=True)
