import os

from src.pdf_loader import load_pdfs
from src.text_chunker import split_documents
from src.embedding_model import load_embedding_model
from src.vector_store import create_vector_store

UPLOAD_FOLDER = "uploaded_pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_uploaded_pdfs(uploaded_files):

    # Remove old PDFs
    for file in os.listdir(UPLOAD_FOLDER):

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file
        )

        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save uploaded PDFs
    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            uploaded_file.name
        )

        with open(file_path, "wb") as file:

            file.write(uploaded_file.getbuffer())

    # Load PDFs
    documents = load_pdfs(UPLOAD_FOLDER)

    # Split into chunks
    chunks = split_documents(documents)

    # Load embedding model
    embedding_model = load_embedding_model()

    # Create vector DB
    vector_db = create_vector_store(
        chunks,
        embedding_model
    )

    return vector_db