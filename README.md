# RAGbot – Retrieval-Augmented AI Agent (CLI)

A command-line Python app that loads local `.txt` files, embeds them using OpenAI, builds a FAISS vector index, and answers questions using GPT-3.5.

Built for fast iteration, agent-ready workflows, and local document intelligence.

---

## Features

- Embeds `.txt` documents using `text-embedding-ada-002`
- Uses FAISS for fast vector search
- Retrieves top-matching chunks
- Sends context to GPT-3.5 for natural language answers
- Terminal-based Q&A loop

---

## Tech Stack

- Python 3.13
- OpenAI API (embeddings + GPT-3.5)
- FAISS for vector similarity search
- `dotenv`, `tqdm`, and basic CLI tooling

---

## Example


