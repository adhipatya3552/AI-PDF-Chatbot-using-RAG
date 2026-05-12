def retrieve_chunks(vector_db, query):

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)

    return docs