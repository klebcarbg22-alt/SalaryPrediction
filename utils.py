from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def process_pdf(file_path):
    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Add metadata (ATS-level improvement)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = file_path
        doc.metadata["page"] = i

    # Split text (optimized)
    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separator="\n"
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings (with cache)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache"
    )

    # Create vector DB
    db = FAISS.from_documents(split_docs, embeddings)

    # Save index
    db.save_local("faiss_index")

    return db
