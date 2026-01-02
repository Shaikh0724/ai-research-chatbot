import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. API Key Load karein
load_dotenv()

def create_vector_db():
    # Folder path jahan aapke 10 papers hain
    DATA_PATH = "data_folder/"
    DB_PATH = "./chroma_db"

    # Agar purana DB hai toh usay clean karein (Safety ke liye)
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        print("--- Purana Database delete kar diya gaya hai ---")

    print(f"--- Papers load ho rahe hain '{DATA_PATH}' se... ---")
    
    # Directory se saari PDFs uthana
    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    print(f"--- Total Pages Load huay: {len(documents)} ---")

    # 2. Chunks mein divide karna (10 papers ke liye optimized settings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"--- Total Chunks banay: {len(chunks)} ---")

    # 3. Embeddings aur Vector Store banan
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("--- Vector Database ban raha hai (ChromaDB)... ---")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"--- Database tayyar hai! Folder: {DB_PATH} ---")

if __name__ == "__main__":
    create_vector_db()