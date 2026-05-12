def create_prompt(context, question):

    return f"""
You are an intelligent AI assistant.

Answer ONLY using the provided context.

Rules:
- Be concise and accurate
- If answer is unavailable, say:
  "I could not find the answer in the document."
- Do not make up information
- Use bullet points when useful

Context:
{context}

Question:
{question}

Answer:
"""