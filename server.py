"""
server.py — FastAPI backend for the local RAG frontend
Run with: uvicorn server:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile, shutil, json, os
from pathlib import Path

from rag.indexer import Indexer
from rag.retriever import Retriever
import config
import ollama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    try:
        ollama.list()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Ask (streaming SSE) ────────────────────────────────────────────────────────


@app.post("/ask")
async def ask(body: dict):
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    retriever = Retriever()
    if not retriever.collection:
        raise HTTPException(
            status_code=400, detail="No documents indexed yet. Upload a document first."
        )

    def stream():
        try:
            # Embed the question
            q_emb = ollama.embeddings(model=config.EMBED_MODEL, prompt=question)[
                "embedding"
            ]

            # Retrieve chunks
            results = retriever.collection.query(
                query_embeddings=[q_emb],
                n_results=config.TOP_K,
                include=["documents", "metadatas", "distances"],
            )

            docs = results["documents"][0]
            distances = results["distances"][0]

            # Filter by similarity
            filtered = [
                doc
                for doc, dist in zip(docs, distances)
                if (1 - dist / 2) >= config.MIN_SIMILARITY
            ]

            if not filtered:
                yield f"data: {json.dumps({'token': 'I could not find relevant information in the indexed documents.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            context = "\n\n---\n\n".join(filtered)
            prompt = (
                "Answer the question using only the context below. "
                "If the context doesn't contain the answer, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

            for chunk in ollama.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                token = chunk["message"]["content"]
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Index ──────────────────────────────────────────────────────────────────────


@app.post("/index")
async def index(
    file: UploadFile = File(None),
    url: str = Query(None),
):
    indexer = Indexer()

    if url:
        try:
            indexer.index_url(url)
            return {"status": "ok", "source": url}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    elif file:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md", ".text"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md",
            )
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            # Rename so metadata stores original filename
            named_path = Path(tempfile.gettempdir()) / file.filename
            shutil.move(tmp_path, named_path)
            indexer.index_file(str(named_path))
            named_path.unlink(missing_ok=True)
            return {"status": "ok", "source": file.filename}
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    else:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a url parameter"
        )


# ── Documents ─────────────────────────────────────────────────────────────────


@app.get("/documents")
async def list_documents():
    try:
        indexer = Indexer()
        meta = indexer.collection.get(include=["metadatas"])["metadatas"]
        sources = {}
        for m in meta:
            s = m.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        return [{"name": k, "chunks": v} for k, v in sorted(sources.items())]
    except Exception:
        return []


@app.delete("/delete")
async def delete_document(document_name: str = Query(...)):
    """Delete a single indexed document by name.

    Args:
        document_name: The filename of the document to delete (e.g., 'example.pdf')
    """
    try:
        if not document_name.strip():
            raise HTTPException(status_code=400, detail="document_name is required")
        indexer = Indexer()
        success = indexer.delete_document(document_name)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Document '{document_name}' not found"
            )
        return {"status": "ok", "deleted": document_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_documents():
    try:
        indexer = Indexer()
        indexer.client.delete_collection(config.COLLECTION_NAME)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Config ────────────────────────────────────────────────────────────────────


@app.get("/config")
async def get_config():
    return {
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "min_similarity": config.MIN_SIMILARITY,
        "chroma_path": config.CHROMA_PATH,
        "collection_name": config.COLLECTION_NAME,
    }


@app.patch("/config")
async def update_config(body: dict):
    mapping = {
        "llm_model": "LLM_MODEL",
        "embed_model": "EMBED_MODEL",
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "top_k": "TOP_K",
        "min_similarity": "MIN_SIMILARITY",
    }
    for key, cfg_key in mapping.items():
        if key in body:
            setattr(config, cfg_key, body[key])
    return {"status": "ok"}


# ── Serve frontend (must be last) ─────────────────────────────────────────────

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
