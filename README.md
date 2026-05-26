# PDF RAG Chatbot

A compact, beginner-friendly Retrieval-Augmented Generation (RAG) project built as a small Python app. Upload a PDF, index its contents, then ask questions through a simple Gradio UI.

## Project overview

- Upload a PDF file.
- Extract text using `PyPDF2`.
- Split the text into overlapping chunks.
- Create embeddings with OpenAI Embeddings API.
- Store embeddings in an in-memory FAISS index.
- Retrieve relevant chunks for a user query and answer using a chat model.

This repository contains a small script (`app.py`) that demonstrates the full workflow for a practical PDF chatbot.

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

- `app.py` — main Gradio script.
- `requirements.txt` — Python dependencies.
- `.gitignore` — files and folders ignored by git.
- `uploads/` — temporary uploads (ignored by git).

## Setup (local / VS Code)

1. Clone the repo and open it in VS Code.
2. Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Set the `OPENAI_API_KEY` environment variable.

Windows Powershell example:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
# Restart your shell after setx
```

Alternatively, you can create a `.env` file and load it in your code.

## How to run the app in VS Code

1. Open the project folder in VS Code.
2. Open the terminal and activate the virtual environment.
3. Run:

```powershell
python app.py
```

4. Open the local Gradio URL shown in the terminal.

## Screenshots

Add screenshots of the running app here for your GitHub README. For example, capture the Gradio UI and a sample Q&A.

## Future improvements

- Persist FAISS index to disk so indexing is faster.
- Use token-aware chunking or sentence-aware splitting.
- Support multiple documents and simple metadata.
- Add tests and CI, and provide a Dockerfile for reproducible runs.

## Notes

This repository is intended as a learning and portfolio project. It is written to be clear and easy to follow rather than production-ready.
