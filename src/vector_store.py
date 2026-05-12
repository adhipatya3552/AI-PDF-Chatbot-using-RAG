from langchain_community.vectorstores import FAISS

def create_vector_store(
    chunks,
    embedding_model
):

    vector_db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_db.save_local("vector_db")

    return vector_db