import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load environment variables and OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# === Load .txt documents from the /docs folder ===
def load_documents(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append({"text": content, "source": file.name})
    return docs

# === Embed docs using TF-IDF ===
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    vectors = vectorizer.fit_transform(texts).toarray()
    for i, doc in enumerate(docs):
        doc["vector"] = vectors[i]
    return docs

# === Find top K matching documents based on cosine similarity ===
def query_index(docs, query, k=3):
    question_vector = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(question_vector, [doc["vector"] for doc in docs])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [docs[i] for i in top_indices]

# === Send prompt to GPT using selected chunks ===
def ask_gpt(question, context_chunks):
    context = "\n\n".join(chunk["text"][:1000] for chunk in context_chunks)
    prompt = f"Answer the question below using only the context provided.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    print("\n=== PROMPT SENT TO GPT ===\n")
    print(prompt[:1000])  # Print first 1000 characters for debugging

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant focused on addiction recovery. Use only the context provided to answer user questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()
