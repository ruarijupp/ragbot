import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === GLOBAL TF-IDF VECTORIZER ===
vectorizer = TfidfVectorizer()

# === LOAD DOCUMENTS ===
def load_documents(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append({"text": content, "source": file.name})
    return docs

# === EMBED WITH TF-IDF ===
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    vectors = vectorizer.fit_transform(texts).toarray()
    for i, doc in enumerate(docs):
        doc["vector"] = vectors[i]
    return docs

# === QUERY SIMILARITY ===
def query_index(docs, query, k=3):
    question_vector = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(question_vector, [doc["vector"] for doc in docs])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [docs[i] for i in top_indices]

# === GPT ANSWER ===
def ask_gpt(question, context_chunks):
    context = "\n\n".join(chunk["text"][:1000] for chunk in context_chunks)
    prompt = f"Answer the question below using only the context provided.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()