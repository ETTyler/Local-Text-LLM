# Local RAG — Ask Questions About Any Document

A fully local, free RAG (Retrieval-Augmented Generation) system.

## How it works

1. You feed it a PDF (a book, paper, manual, anything)
2. It splits the text into chunks and embeds them into a local vector database
3. When you ask a question, it finds the most relevant chunks and passes them to a local LLM
4. The LLM answers using only the content from your document

---

## Setup

### 1. Install Ollama

Download from https://ollama.com and install it for your OS.
Select any model you would like based on your resources and needs. The recommended model is `llama3.1:8b` for best quality, but `gemma:2b` or `phi3` are good lightweight alternatives.
Then pull the models you need:

```bash
ollama pull llama3.1:8b              # The main LLM (~4.7 GB)
ollama pull nomic-embed-text    # The embedding model (~274 MB)
```

Verify Ollama is running:

```bash
ollama list
```

### 2. Set up Python environment

Requires Python 3.9+.

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Index a PDF

```bash
python main.py index path/to/your/book.pdf (or direct link to textfile)
```

This reads the PDF, splits it into chunks, embeds each one, and stores
everything in a local ChromaDB database (`./chroma_db/`).
Only needs to be done once per document.

### Ask a single question

```bash
python main.py ask "What are the main themes of the book?"
python main.py ask "How does the author define consciousness?"
python main.py ask "Summarise chapter 3"
```

### Interactive chat session

```bash
python main.py chat
```

Starts a conversational session where you can ask follow-up questions.
Commands inside chat:

- `sources` — show which passages were used for the last answer
- `clear` — reset conversation history
- `quit` — exit

### Manage your index

```bash
python main.py list    # Show all indexed documents and chunk counts
python main.py clear   # Delete all indexed data and start fresh
```

---

## Configuration

Edit `config.py` to change:

| Setting         | Default            | What it does                             |
| --------------- | ------------------ | ---------------------------------------- |
| `LLM_MODEL`     | `llama3.1:8b`      | Which Ollama model answers questions     |
| `EMBED_MODEL`   | `nomic-embed-text` | Which model creates embeddings           |
| `CHUNK_SIZE`    | `2000`             | Characters per chunk                     |
| `CHUNK_OVERLAP` | `300`              | Overlap between chunks                   |
| `TOP_K`         | `8`                | How many chunks to retrieve per question |

### Lighter model options (lower RAM)

If llama3 is too slow on your machine, edit `config.py`:

```python
LLM_MODEL = "gemma:2b"    # ~1.7 GB, fast
LLM_MODEL = "phi3"        # ~2.3 GB, good quality/speed trade-off
LLM_MODEL = "mistral"     # ~4.1 GB, very capable
```

---

## Project structure

```
local-rag/
├── main.py           # Entry point — CLI commands
├── config.py         # All settings in one place
├── requirements.txt
├── rag/
│   ├── indexer.py    # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py  # Question → similar chunks → LLM → answer
│   └── chat.py       # Interactive multi-turn chat loop
└── chroma_db/        # Created automatically on first index
```

---

## Troubleshooting

**"Connection refused" when running:**
Ollama isn't running. Start it with `ollama serve` or open the Ollama app.

**"No text could be extracted" from a PDF:**
The PDF is likely scanned (image-only). You'll need OCR first — try
`ocrmypdf input.pdf output.pdf` then index the output.

**Answers are slow:**
Try a smaller model (`gemma:2b` or `phi3`). Also try reducing `TOP_K` in config.py.

**Answers are off-topic or wrong:**
Try reducing `CHUNK_SIZE` to 600 and increasing `CHUNK_OVERLAP` to 200,
then re-index. Smaller chunks = more precise retrieval.
