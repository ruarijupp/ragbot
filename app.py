from flask import Flask, request, render_template_string
from ragbot import load_documents, embed_documents, build_index, query_index, ask_gpt

# Load and prepare everything once at startup
docs = load_documents()
embeddings = embed_documents(docs)
index = build_index(embeddings)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>RAGBot – Recovery Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f4f8; padding: 40px; }
        .container { max-width: 700px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        input[type=text], textarea { width: 100%; padding: 12px; margin-top: 10px; border: 1px solid #ccc; border-radius: 5px; }
        button { padding: 10px 20px; background: #2c3e50; color: white; border: none; border-radius: 5px; margin-top: 10px; }
        .response { background: #f7f7f7; padding: 15px; border-radius: 5px; margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAGBot Recovery Chat</h1>
        <form method="post">
            <label for="question">Ask a question related to addiction recovery:</label>
            <input type="text" name="question" id="question" placeholder="e.g. How do I manage cravings?" required>
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
        chunks = query_index(index, docs, question)
        answer = ask_gpt(question, chunks)
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
