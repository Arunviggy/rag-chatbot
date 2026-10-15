
"""Core RAG pipeline: load PDFs -> chunk -> embed -> FAISS -> retrieve -> answer."""

import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from google import genai
from sentence_transformers import SentenceTransformer

# Load .env before reading environment variables
load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"

# Read and clean the Gemini model name
LLM_MODEL = os.getenv(
    "LLM_MODEL", "gemini-3.8-flash"
).strip().strip('"').strip("'")

if LLM_MODEL.startswith("models/"):
    LLM_MODEL = LLM_MODEL.removeprefix("models/")

NOT_FOUND_MSG = "I couldn't find this in the uploaded documents."

_embedder = None


def get_embedder():
    """Load the embedding model once."""
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)

    return _embedder


def load_pdf(file, name):
    """Extract text from each PDF page."""
    reader = PdfReader(file)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({
                "text": text,
                "source": name,
                "page": i,
            })

    return pages


def chunk_pages(pages, chunk_size=800, overlap=100):
    """Split page text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = []

    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({
                "text": piece,
                "source": page["source"],
                "page": page["page"],
            })

    return chunks


def embed(texts):
    """Convert text into normalized embedding vectors."""
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
    )

    return np.asarray(vectors, dtype="float32")


def build_index(chunks):
    """Build a FAISS index using cosine similarity."""
    if not chunks:
        raise ValueError("No text chunks were provided to index.")

    vectors = embed([chunk["text"] for chunk in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def retrieve(question, index, chunks, k=4):
    """Return the most relevant chunks and their similarity scores."""
    if not chunks:
        return []

    question_vector = embed([question])

    scores, ids = index.search(
        question_vector,
        min(k, len(chunks)),
    )

    return [
        (chunks[i], float(score))
        for i, score in zip(ids[0], scores[0])
        if i != -1
    ]


def answer(question, hits, threshold=0.15):
    """Answer using retrieved document context and Gemini."""

    if not hits or hits[0][1] < threshold:
        return NOT_FOUND_MSG

    relevant_hits = [
        (chunk, score)
        for chunk, score in hits
        if score >= threshold
    ]

    if not relevant_hits:
        return NOT_FOUND_MSG

    context = "\n\n".join(
        f"[Source: {chunk['source']}, page {chunk['page']}]\n"
        f"{chunk['text']}"
        for chunk, _ in relevant_hits
    )

    system = (
        "You are a document question-answering assistant. "
        "Answer using ONLY the supplied document context. "
        "Answer clearly and accurately when the information is present. "
        "Cite the source filename and page number, for example "
        "(test_handbook.pdf, p.1). "
        f"If the answer is not in the context, reply exactly: "
        f"{NOT_FOUND_MSG}"
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is missing. "
            "Please check GEMINI_API_KEY in your .env file."
        )

    try:
        client = genai.Client(api_key=api_key)

        # Debug: show the model name in the VS Code terminal
        print("MODEL CHECK:", repr(LLM_MODEL))

        interaction = client.interactions.create(
            model=LLM_MODEL,
            input=(
                f"{system}\n\n"
                f"Document context:\n{context}\n\n"
                f"Question: {question}"
            ),
        )

        return interaction.output_text or NOT_FOUND_MSG

    except Exception as error:
        print(f"Gemini API error: {error}")
        return (
            f"Gemini API error: {error}\n\n"
            "Check the model name, API key, and SDK version."
        )
