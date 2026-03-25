"""
Configuration — edit these to swap models or tune chunking behaviour.
All models listed here run locally via Ollama (free, no API key needed).
"""

# ── Ollama models ──────────────────────────────────────────────────────────────
# The LLM that answers your questions.
# Good options (run `ollama pull <name>` to download):
#   "llama3.1:8b"        ~4.7 GB  — best quality, recommended
#   "mistral"       ~4.1 GB  — fast, very capable
#   "gemma:2b"      ~1.7 GB  — lightweight, runs on low-RAM machines
#   "phi3"          ~2.3 GB  — Microsoft's small but smart model
LLM_MODEL = "llama3.1:8b"

# The embedding model — converts text to vectors for similarity search.
# nomic-embed-text is small (~274 MB) and excellent for this use case.
EMBED_MODEL = "nomic-embed-text"

# ── Chunking settings ──────────────────────────────────────────────────────────
# How many characters per chunk. Larger = more context per chunk but less
# precision. 800–1200 is a good range for most books.
CHUNK_SIZE = 2000

# How many characters chunks overlap. Overlap prevents answers from being split
# across chunk boundaries.
CHUNK_OVERLAP = 300

# ── Retrieval settings ─────────────────────────────────────────────────────────
# How many chunks to retrieve and send as context to the LLM.
# More = richer context, but slower and uses more of the model's context window.
TOP_K = 8

# Minimum similarity score (0–1) for a chunk to be included.
# Lower = more chunks returned but less relevant. 0.3 is a sensible default.
MIN_SIMILARITY = 0.0

# ── Storage ────────────────────────────────────────────────────────────────────
# Where ChromaDB stores its data on your machine.
CHROMA_PATH = "./chroma_db"

# ChromaDB collection name — think of it as a "namespace" for your documents.
COLLECTION_NAME = "local_rag"

# ── Ollama connection ──────────────────────────────────────────────────────────
# Change this if you're running Ollama on a different port or remote machine.
OLLAMA_BASE_URL = "http://localhost:11434"
