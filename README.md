# PDF RAG Chatbot (Notebook)

A compact, beginner-friendly Retrieval-Augmented Generation (RAG) project implemented as a Jupyter notebook. Upload a PDF, index its contents, then ask questions through a simple Gradio UI.

## Project overview

- Upload a PDF file.
- Extract text using `PyPDF2`.
- Split the text into overlapping chunks.
- Create embeddings with OpenAI Embeddings API.
- Store embeddings in an in-memory FAISS index.
- Retrieve relevant chunks for a user query and answer using a chat model.

This repository contains both a small script (`app.py`) and a more exploratory, documented notebook `PDF_RAG_Chatbot.ipynb` intended for learning and demonstration.

## Technologies used

- Python 3.8+
- Gradio for UI
- OpenAI (Embeddings + ChatCompletion)
- FAISS for vector search
- PyPDF2 for PDF text extraction

## RAG workflow (short)

RAG augments a language model with retrieved context at query time. Steps:

1. Split document text into chunks and convert each chunk into a vector embedding.
2. Store embeddings in a vector index (FAISS).
3. For each user question, find nearest chunks and provide them as context to the LLM.

This keeps prompts focused and reduces the amount of text sent to the model.

## Folder structure

- `PDF_RAG_Chatbot.ipynb` — main notebook with step-by-step code and explanations.
- `app.py` — small Gradio script (quick demo).
- `requirements.txt` — Python dependencies.
- `.gitignore` — files and folders ignored by git.
- `uploads/` — temporary uploads (ignored by git).

## Setup (local / VS Code)

1. Clone the repo and open it in VS Code.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set the `OPENAI_API_KEY` environment variable.

Windows Powershell example:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
# Restart your shell after setx
```

Alternatively, you can create a `.env` file and load it in the notebook or script.

## How to run the notebook in VS Code

1. Open `PDF_RAG_Chatbot.ipynb` in VS Code (it supports native notebook editing).
2. Run cells in order. When you reach the Gradio cell, it will show a local URL to interact with the UI.

## Screenshots

Include screenshots of the running app in this section for your GitHub README. For example, capture the Gradio UI and a sample Q&A.

## Future improvements

- Persist FAISS index to disk so indexing is faster.
- Use token-aware chunking or sentence-aware splitting.
- Support multiple documents and simple metadata.
- Add tests and CI, and provide a Dockerfile for reproducible runs.

## Notes

This notebook is intended as a learning/portfolio piece. It is written to be clear and approachable rather than production-ready.
