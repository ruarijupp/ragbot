from flask import Flask, request, render_template_string
from ragbot import load_documents, embed_documents, query_index, ask_gpt
import os

# === Load once ===
docs = load_documents()
docs = embed_documents(docs)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>RAGBot Recovery Chat</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f9fbfc;
            display: flex;
            justify-content: center;
            padding-top: 80px;
        }
        .box {
            background: #ffffff;
            border-radius: 12px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 0 20px rgba(0,0,0,0.05);
        }
        h1 {
            font-size: 28px;
            color: #34495e;
            margin-bottom: 10px;
        }
        label {
            font-size: 16px;
            color: #2c3e50;
        }
        input[type="text"] {
            width: 100%;
            padding: 14px;
            font-size: 16px;
            margin-top: 10px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        button {
            margin-top: 20px;
            padding: 12px 24px;
            font-size: 16px;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .response {
            margin-top: 30px;
            background: #f4f6f7;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>RAGBot Recovery Chat</h1>
        <form method="POST">
            <label for="question">Ask a question related to addiction recovery:</label>
            <input type="text" name="question" placeholder="e.g. How do I manage cravings?" required>
            <button type="submit">Ask</button>
        </form>
        {% if answer %}
        <div class="response">
            <strong>Answer:</strong><br>{{ answer }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = None
    if request.method == "POST":
        question = request.form["question"]
        try:
            chunks = query_index(docs, question)
            answer = ask_gpt(question, chunks)
        except Exception as e:
            answer = f"⚠️ An error occurred: {str(e)}"
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
