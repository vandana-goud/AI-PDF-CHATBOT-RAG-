import os
import tempfile
from typing import List

import numpy as np
import faiss
import openai
import PyPDF2
import gradio as gr



INDEX = None
CHUNKS: List[str] = []


def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF file path."""
    try:
        reader = PyPDF2.PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks (by characters)."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API for a list of texts.

    Requires `OPENAI_API_KEY` environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    openai.api_key = api_key
    try:
        resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
        return [d["embedding"] for d in resp["data"]]
    except Exception as e:
        raise RuntimeError(f"Embedding request failed: {e}")


def build_faiss_index(embeddings: List[List[float]]):
    """Create an in-memory FAISS index from embeddings."""
    arr = np.array(embeddings, dtype="float32")
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)
    return index


def process_pdf(file) -> str:
    """Save uploaded file, extract text, chunk, embed and build FAISS index.

    `file` is the object Gradio provides for uploaded files.
    Returns a short status message for the UI.
    """
    global INDEX, CHUNKS
    if file is None:
        return "No file uploaded."
    # Save to a temporary file first
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        # Gradio's uploaded `file` has a `.file` attribute (SpooledTemporaryFile)
        try:
            file.file.seek(0)
            tmp.write(file.file.read())
        except Exception:
            # Fallback: if file is a path string
            if isinstance(file, str):
                tmp.close()
                tmp.name = file
            else:
                tmp.close()
                return "Unsupported file object."
        tmp.close()
        text = extract_text_from_pdf(tmp.name)
        CHUNKS = chunk_text(text)
        if not CHUNKS:
            return "No text found in PDF."
        embeddings = get_embeddings(CHUNKS)
        INDEX = build_faiss_index(embeddings)
        return f"Indexed {len(CHUNKS)} chunks. You can now ask questions."
    except Exception as e:
        return f"Error processing file: {e}"


def retrieve_relevant(question: str, top_k: int = 3) -> List[str]:
    """Return top-k relevant chunks for the question using FAISS nearest neighbor search."""
    global INDEX, CHUNKS
    if INDEX is None:
        return []
    try:
        q_emb = get_embeddings([question])[0]
        xq = np.array([q_emb], dtype="float32")
        distances, indices = INDEX.search(xq, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(CHUNKS):
                results.append(CHUNKS[idx])
        return results
    except Exception:
        return []


def answer_question(question: str) -> str:
    """Compose a prompt with retrieved context and ask OpenAI to answer.

    Basic safety: if no index exists, instruct the user to upload first.
    """
    if not question:
        return "Please enter a question."
    if INDEX is None:
        return "No indexed document. Please upload and index a PDF first."
    try:
        context_chunks = retrieve_relevant(question, top_k=3)
        if not context_chunks:
            return "No relevant context found in the document."
        prompt = "Use the following extracted document passages to answer the question concisely. If the answer is not contained, say you don't know.\n\n"
        for i, c in enumerate(context_chunks, 1):
            prompt += f"[{i}] {c}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY is not set."
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=300)
        return completion["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating answer: {e}"



def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Simple PDF RAG Chatbot\nUpload a PDF, index it, then ask questions.")
        with gr.Row():
            pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
            index_btn = gr.Button("Index PDF")
        status = gr.Textbox(label="Status", interactive=False)
        question = gr.Textbox(label="Ask a question")
        answer = gr.Textbox(label="Answer", interactive=False)

        index_btn.click(fn=process_pdf, inputs=pdf, outputs=status)
        ask_btn = gr.Button("Ask")
        ask_btn.click(fn=answer_question, inputs=question, outputs=answer)
        question.submit(fn=answer_question, inputs=question, outputs=answer)

    demo.launch()


if __name__ == "__main__":
    main()
