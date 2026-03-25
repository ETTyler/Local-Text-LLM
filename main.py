"""
Local RAG (Retrieval-Augmented Generation) — powered by Ollama + ChromaDB
Ask questions about a document completely offline and powered by a local LLM.

Usage:
  python main.py index path/to/book.pdf     # Index a PDF
  python main.py ask "What is X?"           # Ask a question
  python main.py chat                        # Interactive chat session
  python main.py list                        # List indexed documents
  python main.py clear                       # Clear the vector store
"""

import sys
from rag.indexer import Indexer
from rag.retriever import Retriever
from rag.chat import chat_loop


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Usage: python main.py index <path/to/file.pdf>")
            sys.exit(1)
        pdf_path = sys.argv[2]
        indexer = Indexer()
        indexer.index_file(pdf_path)

    elif command == "ask":
        if len(sys.argv) < 3:
            print('Usage: python main.py ask "your question here"')
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        retriever = Retriever()
        retriever.ask(question)

    elif command == "chat":
        chat_loop()

    elif command == "list":
        indexer = Indexer()
        indexer.list_documents()

    elif command == "clear":
        indexer = Indexer()
        indexer.clear()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
