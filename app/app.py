import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '.env'
    )
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Fix: Streamlit's file watcher crashes when it tries to introspect
# torch.classes.__path__._path — patch it to an empty list to suppress
# the RuntimeError without affecting PyTorch functionality.
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st

from src.chat_storage import save_chat, load_chat

from src.pdf_export import export_chat_to_pdf

from src.pdf_processor import process_uploaded_pdfs

from src.retriever import retrieve_chunks

from src.prompt_template import create_prompt

from src.llm_response import generate_response

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.chat-user {
    background-color: #1E293B;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.chat-bot {
    background-color: #111827;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 20px;
    border-left: 4px solid #4F46E5;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 40px;
}

@keyframes fadeOutWarning {
    0%   { opacity: 1; transform: translateY(0); }
    75%  { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-6px); }
}

.timed-warning {
    animation: fadeOutWarning 3s ease-in-out forwards;
    background-color: #2d2200;
    border: 1px solid #facc15;
    border-radius: 8px;
    padding: 12px 16px;
    color: #facc15;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">🤖 AI PDF Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions directly from your PDFs using RAG + LLMs</div>',
    unsafe_allow_html=True
)

# ---------------- TIMED WARNING HELPER ----------------

import streamlit.components.v1 as components

def show_timed_warning(message: str, duration: float = 3.0):
    """Inject a fixed-position toast warning via JS — zero layout impact."""
    duration_ms = int(duration * 1000)
    # Escape message for safe JS string injection
    safe_msg = message.replace("'", "\\'").replace('"', '&quot;')
    components.html(
        f"""
        <script>
        (function() {{
            var toast = parent.document.createElement('div');
            toast.innerHTML = '<span style="font-size:16px">&#9888;&#65039;</span> {safe_msg}';
            Object.assign(toast.style, {{
                position:     'fixed',
                bottom:       '28px',
                right:        '28px',
                background:   '#2d2200',
                border:       '1px solid #facc15',
                borderRadius: '10px',
                padding:      '12px 18px',
                color:        '#facc15',
                fontSize:     '14px',
                fontFamily:   'sans-serif',
                zIndex:       '99999',
                boxShadow:    '0 6px 24px rgba(0,0,0,0.5)',
                display:      'flex',
                alignItems:   'center',
                gap:          '8px',
                opacity:      '1',
                transition:   'opacity 0.6s ease, transform 0.6s ease',
                transform:    'translateY(0)',
            }});
            parent.document.body.appendChild(toast);
            // Fade-out near the end
            setTimeout(function() {{
                toast.style.opacity   = '0';
                toast.style.transform = 'translateY(8px)';
            }}, {duration_ms} - 600);
            // Remove from DOM after full duration
            setTimeout(function() {{
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }}, {duration_ms});
        }})();
        </script>
        """,
        height=0,
    )

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ---------------- SIDEBAR ----------------

with st.sidebar:

    # ---------------- PDF UPLOAD ----------------

    st.subheader("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚡ Process PDFs"):

        if uploaded_files:

            with st.spinner("Processing PDFs..."):

                vector_db = process_uploaded_pdfs(
                    uploaded_files
                )

                st.session_state.vector_db = vector_db

            st.success(
                "PDFs processed successfully!"
            )

        else:
            show_timed_warning(
                "Please upload at least one PDF."
            )

    st.divider()

    # ---------------- CLEAR CHAT ----------------

    if st.button("🗑️ Clear Chat History"):

        if len(st.session_state.messages) == 0:

            show_timed_warning("No chat history to clear.")

        else:

            st.session_state.messages = []

            save_chat([])

            st.rerun()

    st.divider()

    # ---------------- DOWNLOAD CHAT ----------------

    if len(st.session_state.messages) == 0:

        if st.button("⬇ Download Chat"):

            show_timed_warning(
                "No chat history available to download."
            )

    else:

        chat_text = ""

        for message in st.session_state.messages:

            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            chat_text += (
                f"{role}: {message['content']}\n\n"
            )

        st.download_button(
            label="⬇ Download Chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

    st.divider()

    # ---------------- EXPORT PDF ----------------

    if len(st.session_state.messages) == 0:

        if st.button("📄 Export Chat as PDF"):

            show_timed_warning(
                "No chat history available to export."
            )

    else:

        pdf_data = export_chat_to_pdf(
            st.session_state.messages
        )

        st.download_button(
            label="📄 Export Chat as PDF",
            data=pdf_data,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )

# ---------------- USER INPUT ----------------

query = st.chat_input(
    "Ask something from the PDF..."
)

if query:

    # Check if PDFs processed
    if st.session_state.vector_db is None:

        show_timed_warning(
            "Please upload and process PDFs first."
        )

        st.stop()

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.spinner("Thinking..."):

        docs = retrieve_chunks(
            st.session_state.vector_db,
            query
        )

        context = ""

        for i, doc in enumerate(docs):

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            context += f"""
            Source {i+1} (Page {page}):

            {doc.page_content}

            """

        prompt = create_prompt(
            context,
            query
        )

        response = generate_response(
            prompt
        )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Save locally
    save_chat(
        st.session_state.messages
    )

# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f'''
            <div class="chat-user">
            🧑 {message["content"]}
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'''
            <div class="chat-bot">
            🤖 {message["content"]}
            </div>
            ''',
            unsafe_allow_html=True
        )