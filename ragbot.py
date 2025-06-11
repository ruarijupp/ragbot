import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# === Load environment and OpenAI client ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Global TF-IDF vectorizer ===
vectorizer = TfidfVectorizer()

# === Load documents from /docs folder ===
def load_documents(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append({"text": content, "source": file.name})
    return docs

# === Embed documents using TF-IDF ===
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    vectors = vectorizer.fit_transform(texts).toarray()
    for i, doc in enumerate(docs):
        doc["vector"] = vectors[i]
    return docs

# === Query docs and return top matches + top score ===
def query_index(docs, query, k=3):
    question_vector = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(question_vector, [doc["vector"] for doc in docs])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    top_score = similarities[top_indices[0]] if top_indices.size > 0 else 0
    return top_docs, top_score

# === Ask GPT using retrieved context chunks ===
def ask_gpt(question, context_chunks):
    context = "\n\n".join(chunk["text"][:1000] for chunk in context_chunks)
    prompt = f"Answer the question below using only the context provided.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    print("\n=== GPT Prompt (RAG mode) ===\n")
    print(prompt[:1000])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a compassionate recovery assistant. Use only the provided context to answer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# === Fallback if no doc is a good match ===
def ask_general_gpt(question):
    print("\n=== GPT Prompt (General mode) ===\n")
    print(question)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant supporting individuals in addiction recovery. Respond with empathy, clarity, and encouragement."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()
