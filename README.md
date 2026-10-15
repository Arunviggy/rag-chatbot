# PDF RAG Chatbot

A simple PDF Question Answering chatbot built using Python, Streamlit, FAISS, Sentence Transformers, and Google Gemini.

## Features
- Upload PDF documents
- Extract text from PDFs
- Split text into chunks
- Search relevant information using FAISS
- Generate answers using Google Gemini
- Display document sources and page numbers

## Technologies Used
- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini API
- PyPDF

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual API key. Never upload your real API key to GitHub.

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

- `app.py` — Streamlit user interface
- `rag.py` — PDF processing, retrieval, and answer generation
- `requirements.txt` — Python dependencies
- `check_models.py` — Model checking script

## Future Improvements
- Improve response speed
- Add OCR for scanned PDFs
- Improve source citations
