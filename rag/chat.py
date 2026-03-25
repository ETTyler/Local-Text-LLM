"""
Chat
----
Interactive multi-turn chat session over your indexed documents.
Maintains conversation history so you can ask follow-up questions naturally.

Type 'quit' or 'exit' to leave.
Type 'sources' to see which chunks were used for the last answer.
Type 'clear' to reset conversation history.
"""

from rag.retriever import Retriever
import config


def chat_loop():
    print("\n── Local RAG Chat ──────────────────────────────────────────")
    print(f"  LLM     : {config.LLM_MODEL}")
    print(f"  Embedder: {config.EMBED_MODEL}")
    print(f"  Store   : {config.CHROMA_PATH}")
    print("  Commands: 'sources', 'clear', 'quit'")
    print("────────────────────────────────────────────────────────────\n")

    retriever = Retriever()
    if not retriever.collection:
        return

    history = []
    last_question = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            last_question = None
            print("[conversation history cleared]\n")
            continue

        if user_input.lower() == "sources":
            if not last_question:
                print("No question asked yet.\n")
            else:
                sources = retriever.get_sources(last_question)
                print("\nSources used:")
                for s in sources:
                    print(f"  [{s['score']:.2f}] {s['source']} chunk {s['chunk']}: {s['text']}")
                print()
            continue

        last_question = user_input
        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        answer = retriever.ask(user_input)
        print()

        history.append({"role": "assistant", "content": answer})
