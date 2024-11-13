import pickle
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

def preprocess_data(openai_api_key, urls_file="unique_links_list.txt"):
    """
    Preprocess URLs and save as a single pickle file
    """
    print("Starting preprocessing...")
    
    # Set up OpenAI API key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
    # Load URLs
    print("Loading URLs...")
    with open(urls_file, 'r') as file:
        urls = [line.strip() for line in file.readlines()]
    
    # Load and process documents
    print("Processing documents...")
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    
    # Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in splits]
    embedded_texts = embeddings.embed_documents(texts)
    
    # Save everything in a single file
    print("Saving processed data...")
    data_to_save = {
        'texts': texts,
        'embeddings': embedded_texts,
        'metadata': [doc.metadata for doc in splits]
    }
    
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print("Preprocessing complete!")
    return data_to_save

if __name__ == "__main__":
    api_key = input("Enter your OpenAI API key: ")
    preprocessed_data = preprocess_data(api_key)
