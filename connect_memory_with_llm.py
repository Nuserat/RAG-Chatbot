from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

class DocumentProcessor:
    """Process and vectorize academic documents from multiple formats"""
    
    def __init__(self, data_path="Doc", db_path="vectorstore/db_faiss"):
        self.data_path = data_path
        self.db_path = db_path
        self.documents = []
        self.text_chunks = []
        self.embedding_model = None
        self.db = None
    
    def load_documents(self):
        """Load PDF, JSON, and CSV files from data directory"""
        print("📄 Loading documents...")
        
        try:
            if any(Path(self.data_path).glob("*.pdf")):
                pdf_loader = DirectoryLoader(
                    self.data_path,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader,
                )
                pdf_docs = pdf_loader.load()
                self.documents.extend(pdf_docs)
                print(f"✓ PDFs loaded: {len(pdf_docs)} documents")
        except Exception as e:
            print(f"⚠ PDF loading error: {e}")
        
        print(f"📊 Total documents loaded: {len(self.documents)}")
        return self.documents
    
    def create_chunks(self):
        """Split documents into overlapping chunks for better context"""
        print("\n✂️  Creating text chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"📦 Total chunks created: {len(self.text_chunks)}")
        return self.text_chunks
    
    def load_embedding_model(self):
        """Load HuggingFace sentence transformer for embeddings"""
        print("\n🤖 Loading embedding model...")
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Embedding model loaded successfully")
            return self.embedding_model
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def create_vectorstore(self):
        """Create FAISS vector store from document chunks"""
        print("\n🔍 Creating FAISS vector store...")
        
        try:
            self.db = FAISS.from_documents(self.text_chunks, self.embedding_model)
            os.makedirs(self.db_path, exist_ok=True)
            self.db.save_local(self.db_path)
            print(f"✓ Vector store created and saved at: {self.db_path}")
            return self.db
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def process(self):
        """Execute complete document processing pipeline"""
        try:
            print("=" * 60)
            print("🚀 Starting Document Processing Pipeline")
            print("=" * 60)
            
            self.load_documents()
            self.create_chunks()
            self.load_embedding_model()
            self.create_vectorstore()
            
            print("\n" + "=" * 60)
            print("✅ Document processing completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            raise

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process()
