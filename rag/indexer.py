"""
Indexer
-------
Handles loading a PDF, splitting it into chunks, embedding each chunk
with Ollama (nomic-embed-text), and storing everything in ChromaDB.

Run once per document. Re-indexing the same file is safely skipped.
"""

import os
import tempfile
import urllib.request
from urllib.parse import urlparse
import hashlib
from pathlib import Path

import pdfplumber
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

import config


class Indexer:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_file(self, file_path: str) -> None:
        """Accept a local file path or a URL — auto-detects both."""
        if file_path.startswith("http://") or file_path.startswith("https://"):
            self.index_url(file_path)
            return

        path = Path(file_path)
        if not path.exists():
            print(f"[error] File not found: {file_path}")
            return

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            self._index(path, self._extract_pdf_text(path), file_type="PDF")
        elif suffix in (".txt", ".text", ".md"):
            self._index(path, self._extract_text_file(path), file_type="text")
        else:
            print(
                f"[error] Unsupported file type '{suffix}'. Supported: .pdf, .txt, .md"
            )

    def index_url(self, url: str) -> None:
        """Download a .txt or .pdf from a URL and index it."""
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "download"
        suffix = Path(parsed.path).suffix.lower()

        if suffix not in (".txt", ".text", ".md", ".pdf", ""):
            print(
                f"[error] Unsupported URL file type '{suffix}'. Supported: .pdf, .txt, .md"
            )
            return

        print(f"[download] Fetching: {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw_bytes = response.read()
                content_type = response.headers.get("Content-Type", "")
        except Exception as e:
            print(f"[error] Could not download URL: {e}")
            return

        # Detect PDF by content-type or magic bytes even if URL has no extension
        is_pdf = (
            suffix == ".pdf"
            or "application/pdf" in content_type
            or raw_bytes[:4] == b"%PDF"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            if is_pdf:
                tmp_path = Path(tmpdir) / (
                    filename if suffix == ".pdf" else filename + ".pdf"
                )
                tmp_path.write_bytes(raw_bytes)
                # Use the original URL filename as the display name in metadata
                display_path = Path(tmpdir) / filename
                display_path.write_bytes(raw_bytes)
                self._index(
                    display_path,
                    self._extract_pdf_text(tmp_path),
                    file_type="PDF (URL)",
                )
            else:
                # Plain text — decode with fallback
                text = None
                for encoding in ("utf-8", "utf-8-sig", "latin-1"):
                    try:
                        text = raw_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if text is None:
                    print("[error] Could not decode downloaded content as text.")
                    return
                tmp_path = Path(tmpdir) / filename
                tmp_path.write_text(text, encoding="utf-8")
                self._index(tmp_path, text, file_type="text (URL)")

    def index_pdf(self, pdf_path: str) -> None:
        """Kept for backwards compatibility — delegates to index_file."""
        self.index_file(pdf_path)

    def _index(self, path: Path, text: str, file_type: str) -> None:
        """Shared indexing logic for any file type once text is extracted."""
        file_id = self._file_id(path)

        existing = self.collection.get(where={"source": str(path.name)}, limit=1)
        if existing["ids"]:
            print(f"[skip] '{path.name}' is already indexed. Use `clear` to re-index.")
            return

        if not text.strip():
            print(f"[error] No text could be extracted from '{path.name}'.")
            return

        print(f"[1/4] Read {file_type} file: {path.name} ({len(text):,} characters)")

        print(
            f"[2/4] Splitting into chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})"
        )
        chunks = self.splitter.split_text(text)
        print(f"       Created {len(chunks)} chunks")

        print(f"[3/4] Embedding chunks with '{config.EMBED_MODEL}' via Ollama...")
        embeddings = self._embed_chunks(chunks)

        print(f"[4/4] Storing in ChromaDB at '{config.CHROMA_PATH}'")
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": path.name, "chunk_index": i} for i in range(len(chunks))
        ]

        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"       Stored chunks {start}–{min(end, len(chunks))}/{len(chunks)}")

        print(f"\n✓ Done! '{path.name}' is ready to query.")
        print(f'  Run: python main.py ask "your question here"')

    def list_documents(self) -> None:
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        if not all_meta:
            print("No documents indexed yet.")
            return
        sources = sorted(set(m["source"] for m in all_meta))
        print(f"\nIndexed documents ({len(sources)}):")
        for s in sources:
            count = sum(1 for m in all_meta if m["source"] == s)
            print(f"  • {s}  ({count} chunks)")

    def clear(self) -> None:
        confirm = input(
            "This will delete all indexed documents. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() == "yes":
            self.client.delete_collection(config.COLLECTION_NAME)
            print("✓ Vector store cleared.")
        else:
            print("Aborted.")

    # ── Internals ──────────────────────────────────────────────────────────────

    def _extract_pdf_text(self, path: Path) -> str:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages)

    def _extract_text_file(self, path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode '{path.name}' — unknown encoding.")

    def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        embeddings = []
        for i, chunk in enumerate(chunks):
            print(
                f"\r       Embedding chunk {i + 1}/{len(chunks)}...", end="", flush=True
            )
            response = ollama.embeddings(model=config.EMBED_MODEL, prompt=chunk)
            embeddings.append(response["embedding"])
        print()  # newline after progress
        return embeddings

    @staticmethod
    def _file_id(path: Path) -> str:
        """Short stable ID derived from filename + size."""
        raw = f"{path.name}_{path.stat().st_size}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
