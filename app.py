
import streamlit as st
from dotenv import load_dotenv

# Load environment variables BEFORE importing rag
load_dotenv()

import rag


# ---------- Page configuration ----------
st.set_page_config(
    page_title="Document Q&A (RAG)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with your documents")
st.caption("Upload PDFs and ask questions based on their contents.")


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=200,
        max_value=1500,
        value=800,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=25,
    )

    top_k = st.slider(
        "Chunks retrieved (k)",
        min_value=1,
        max_value=8,
        value=4,
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=0.8,
        value=0.15,
        step=0.05,
    )

    st.caption(
        "Lower thresholds allow less-similar chunks. "
        "Very low values may produce irrelevant answers."
    )

    st.divider()

    files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", disabled=not files):
        with st.spinner("Reading, chunking, and embedding PDFs..."):
            try:
                pages = []

                for pdf_file in files:
                    pages.extend(
                        rag.load_pdf(pdf_file, pdf_file.name)
                    )

                chunks = rag.chunk_pages(
                    pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                if not chunks:
                    st.error(
                        "No readable text was found. "
                        "Scanned PDFs may need OCR."
                    )
                else:
                    index = rag.build_index(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.messages = []

                    st.success(
                        f"Successfully indexed {len(chunks)} chunks "
                        f"from {len(files)} PDF file(s)."
                    )

            except Exception as e:
                st.error(f"Could not process the PDFs: {e}")


# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- Ask a question ----------
question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:
    if "index" not in st.session_state:
        st.warning(
            "Please upload your PDF documents and click "
            "'Process documents' first."
        )
        st.stop()

    # Display user question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks and generate an answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                hits = rag.retrieve(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=top_k,
                )

                reply = rag.answer(
                    question,
                    hits,
                    threshold=threshold,
                )

                st.markdown(reply)

                # Display retrieved sources
                with st.expander("📚 Sources (retrieved chunks)"):
                    if hits:
                        for chunk, score in hits:
                            st.markdown(
                                f"**{chunk['source']}, "
                                f"page {chunk['page']}** "
                                f"(similarity: {score:.3f})"
                            )

                            st.caption(
                                chunk["text"][:500]
                                + ("..." if len(chunk["text"]) > 500 else "")
                            )
                    else:
                        st.write("No matching chunks were retrieved.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )

            except Exception as e:
                st.error(
                    f"An error occurred while answering your question: {e}"
                )
