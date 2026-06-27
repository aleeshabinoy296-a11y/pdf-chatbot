# PDF Chatbot

A GenAI web app that lets you upload any PDF and ask questions about it in plain English. Powered by LangChain, OpenAI GPT-3.5, and FAISS vector search.

## Demo

![App screenshot](screenshot.png)

Live app: [Add your deployed Streamlit link here once deployed]

## What it does

- Upload any PDF (textbook, research paper, legal document, manual, report)
- Ask questions about the content in plain English
- The app finds the most relevant sections and uses GPT to answer
- Full conversation memory — ask follow-up questions naturally
- Shows the exact PDF excerpts used to generate each answer (so you can verify)
- Clear chat button to start a fresh conversation

## Tech stack

- Python
- Streamlit — for the web interface
- LangChain — orchestrates the retrieval + generation pipeline
- OpenAI GPT-3.5-turbo — generates natural language answers
- OpenAI Embeddings (text-embedding-ada-002) — converts text chunks to vectors
- FAISS — vector store for fast similarity search
- PyPDF2 — extracts text from PDF files
- tiktoken — token counting for OpenAI API

## How it works (RAG pipeline)

This app uses a technique called **Retrieval Augmented Generation (RAG)**:

1. **Extract** — PyPDF2 reads all text from the uploaded PDF
2. **Split** — LangChain splits the text into overlapping chunks of ~1000 characters
3. **Embed** — each chunk is converted into a numerical vector (embedding) using OpenAI's embedding model
4. **Store** — all vectors are stored in a FAISS index (an in-memory vector database)
5. **Retrieve** — when you ask a question, it is also embedded and the 4 most similar chunks are retrieved
6. **Generate** — the question + retrieved chunks are sent to GPT-3.5, which answers using only that context
7. **Remember** — LangChain's ConversationBufferMemory keeps the chat history so follow-up questions work naturally

RAG is one of the most important patterns in production AI systems today. It allows a language model to answer questions about documents it has never seen before, without retraining.

## How to run this locally

1. Clone this repository
   ```
   git clone https://github.com/YOUR-USERNAME/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. Install the required packages
   ```
   pip install -r requirements.txt
   ```

3. Get an OpenAI API key
   - Go to platform.openai.com
   - Create an account and add a small amount of credit (this app uses ~$0.01 per conversation)
   - Copy your API key (starts with `sk-`)

4. Run the app
   ```
   streamlit run app.py
   ```

5. Paste your API key into the sidebar, upload a PDF, and start chatting

## What I learned building this

- How Retrieval Augmented Generation (RAG) works end to end
- Text chunking strategies and why chunk overlap matters (so answers don't get split at chunk boundaries)
- How vector embeddings represent meaning as numbers (similar meaning = similar vectors = close in vector space)
- FAISS — a fast vector similarity search library developed by Meta
- LangChain's ConversationalRetrievalChain — chaining retrieval + generation + memory together
- Managing Streamlit session state to preserve the model and chat history across reruns

## Limitations

- Requires an OpenAI API key (small cost per use, typically under ₹1 per conversation)
- Cannot read scanned PDFs or image-based PDFs (only text-layer PDFs work)
- The FAISS index is built fresh each time a new PDF is uploaded — for very large PDFs (100+ pages) this takes 10–20 seconds
- GPT answers are based only on retrieved chunks, so if the answer is not in the PDF, it will say it doesn't know

## Possible improvements

- Support scanned PDFs using OCR (pytesseract)
- Let users upload multiple PDFs and search across all of them
- Add a local model option (e.g. Ollama + LLaMA) so it works without an API key
- Save the FAISS index to disk so it doesn't need to be rebuilt on every session
- Add a source page number indicator so users can find the exact page the answer came from
