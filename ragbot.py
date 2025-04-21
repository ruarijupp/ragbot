import os
import faiss
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and chunk documents
def load_documents(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append({"text": content, "source": file.name})
    return docs

# Embed documents with OpenAI (new SDK)
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = []

    for i in tqdm(range(0, len(texts), 20), desc="Embedding"):
        batch = texts[i:i+20]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        for e in response.data:
            embeddings.append(e.embedding)

    return np.array(embeddings).astype("float32")

# Build FAISS index
def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Query top-k similar chunks (new SDK)
def query_index(index, docs, query, k=3):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding
    query_vector = np.array([query_embedding]).astype("float32")
    D, I = index.search(query_vector, k)
    return [docs[i] for i in I[0]]

# Ask GPT with context (new SDK)
def ask_gpt(question, context_chunks):
    context = "\n\n".join(chunk["text"][:1000] for chunk in context_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQ: {question}\nA:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Loading documents...")
    docs = load_documents()

    if not docs:
        print("❌ No documents found in /docs. Add .txt files and try again.")
        exit()

    print("📐 Creating embeddings...")
    embeddings = embed_documents(docs)

    print("📚 Building index...")
    index = build_index(embeddings)

    print("\n🤖 Ask your question (type 'exit' to quit):")
while True:
    q = input("\n> ").strip()
    if q.lower() in ["exit", "quit"]:
        break
    if not q:
        print("⚠️  Please enter a question.")
        continue

    top_chunks = query_index(index, docs, q)
    answer = ask_gpt(q, top_chunks)
    print(f"\n🧠 Answer: {answer}")


