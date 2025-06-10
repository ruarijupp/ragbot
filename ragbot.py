import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. Load Documents
def load_documents():
    folder = "data"
    if not os.path.exists(folder) or not os.listdir(folder):
        print("⚠️ No documents found. Loading fallback content.")
        return ["Addiction is a chronic condition that can be treated with support, structure, and healthy routines."]
    
    documents = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), "r") as f:
            documents.append(f.read())
    return documents

# 2. Embed Documents (TF-IDF)
def embed_documents(documents):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(tqdm(documents, desc="Embedding"))
    return embeddings

# 3. Build Index (returns FAISS-like format)
def build_index(embeddings):
    vectors = embeddings.toarray()
    dim = len(vectors[0])
    return {"vectors": vectors, "dim": dim}

# 4. Query Index
def query_index(index, documents, question):
    question_vector = TfidfVectorizer().fit(documents + [question]).transform([question]).toarray()[0]
    similarities = cosine_similarity([question_vector], index["vectors"])[0]
    top_indices = np.argsort(similarities)[::-1][:3]
    return [documents[i] for i in top_indices]

# 5. Ask GPT with Retrieved Docs
def ask_gpt(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a recovery chatbot. Use the information below to answer the user's question in a compassionate, supportive tone.

Context:
{context}

User: {question}
Bot:"""

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
