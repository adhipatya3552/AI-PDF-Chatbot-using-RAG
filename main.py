from pathlib import Path

from src.pdf_loader import load_pdfs
from src.text_chunker import split_documents
from src.embedding_model import load_embedding_model
from src.vector_store import create_vector_store


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploaded_pdfs"
DATA_FOLDER = BASE_DIR / "data"


def build_vector_database(pdf_folder: Path):
    """Load PDFs, split them, create embeddings, and save FAISS index."""
    print(f"Looking for PDFs in: {pdf_folder}")

    if not pdf_folder.exists():
        raise FileNotFoundError(f"Folder not found: {pdf_folder}")

    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {pdf_folder}\n"
            f"Add PDF files to this folder and run the script again."
        )

    documents = load_pdfs(str(pdf_folder))
    print(f"Documents loaded: {len(documents)}")

    if not documents:
        raise ValueError("No text could be extracted from the PDF files.")

    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created. Check the PDF content and chunking logic.")

    embedding_model = load_embedding_model()
    print("Embedding model loaded successfully.")

    vector_db = create_vector_store(chunks, embedding_model)
    print("Vector database created successfully and saved to ./vector_db")

    return vector_db


if __name__ == "__main__":
    # Prefer uploaded_pdfs for your current Streamlit workflow.
    # If it is empty, fall back to data/ for manual testing.
    try:
        if UPLOAD_FOLDER.exists() and list(UPLOAD_FOLDER.glob("*.pdf")):
            build_vector_database(UPLOAD_FOLDER)
        else:
            build_vector_database(DATA_FOLDER)
    except Exception as e:
        print(f"Error: {e}")