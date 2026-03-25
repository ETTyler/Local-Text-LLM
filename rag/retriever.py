"""
Retriever
---------
Given a question:
  1. Embed the question with the same model used during indexing
  2. Find the top-k most similar chunks in ChromaDB
  3. Build a grounded prompt and send it to the local LLM via Ollama
  4. Stream the answer back to the terminal
"""

from chromadb.config import Settings
import chromadb
import ollama

import config


SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context passages.

Rules:
- Only use information from the context below to answer.
- If the context does not contain enough information, say "I couldn't find that in the document."
- Be concise but complete. Quote relevant passages when helpful.
- Never make up or infer facts not present in the context.
"""

PROMPT_TEMPLATE = """Context passages from the document:
{context}

---
Question: {question}

Answer based only on the context above:"""


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
        except Exception:
            print("[error] No documents indexed yet. Run: python main.py index <path/to/file.pdf>")
            self.collection = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str, stream: bool = True) -> str:
        if not self.collection:
            return ""

        # 1. Embed the question
        q_embedding = ollama.embeddings(
            model=config.EMBED_MODEL,
            prompt=question,
        )["embedding"]

        # 2. Retrieve top-k chunks
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=config.TOP_K,
            include=["documents", "distances", "metadatas"],
        )

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Filter by similarity threshold (ChromaDB cosine distance: 0=identical, 2=opposite)
        # Convert distance to similarity: similarity = 1 - (distance / 2)
        filtered = [
            (doc, meta, 1 - dist / 2)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if (1 - dist / 2) >= config.MIN_SIMILARITY
        ]

        if not filtered:
            answer = "I couldn't find relevant information in the indexed documents for that question."
            print(answer)
            return answer

        # 3. Build context string
        context_parts = []
        for i, (doc, meta, score) in enumerate(filtered):
            source = meta.get("source", "unknown")
            chunk_idx = meta.get("chunk_index", "?")
            context_parts.append(
                f"[Passage {i+1} | source: {source}, chunk: {chunk_idx}, relevance: {score:.2f}]\n{doc}"
            )
        context = "\n\n".join(context_parts)

        # 4. Ask the LLM
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        if stream:
            return self._stream_answer(prompt)
        else:
            return self._get_answer(prompt)

    def get_sources(self, question: str) -> list[dict]:
        """Return the source chunks used without generating an answer — useful for debugging."""
        if not self.collection:
            return []
        q_embedding = ollama.embeddings(model=config.EMBED_MODEL, prompt=question)["embedding"]
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=config.TOP_K,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {
                "text": doc[:200] + "...",
                "source": meta.get("source"),
                "chunk": meta.get("chunk_index"),
                "score": round(1 - dist / 2, 3),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _stream_answer(self, prompt: str) -> str:
        full_response = ""
        stream = ollama.chat(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token
        print()  # trailing newline
        return full_response

    def _get_answer(self, prompt: str) -> str:
        response = ollama.chat(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
