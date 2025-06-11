import os
from flask import Flask, request, render_template_string
from ragbot import load_documents, embed_documents, query_index, ask_gpt, ask_general_gpt

app = Flask(__name__)

# === Lazy load and embed only once ===
docs = []

def get_docs():
    global docs
    if not docs:
        raw = load_documents()
        docs = embed_documents(raw)
    return docs

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Umatilla County Housing Authority</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f9fbfc;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #2C3E50;
            padding: 20px 40px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-left {
            display: flex;
            align-items: center;
        }
        .header-left img {
            height: 120px;
            margin-right: 20px;
        }
        .header-left h2 {
            margin: 0;
        }
        nav a {
            margin: 0 15px;
            color: #ecf0f1;
            text-decoration: none;
            font-weight: bold;
        }
        .container {
            display: flex;
            justify-content: center;
            padding: 40px 20px 20px;
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
        .thought {
            margin: 50px auto 30px;
            text-align: center;
            font-style: italic;
            color: #2c3e50;
            font-size: 18px;
            max-width: 700px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <img src="/static/ucha-logo.png" alt="UCHA Logo" />
            <h2>Umatilla County Housing Authority</h2>
        </div>
        <nav>
            <a href="#">Home</a>
            <a href="#">Thought of the Week</a>
            <a href="#">Resources</a>
            <a href="#">About</a>
        </nav>
    </header>

    <div class="container">
        <div class="box">
            <h1>Recovery Assistant</h1>
            <form method="POST">
                <label for="question">Ask a question related to addiction recovery:</label>
                <input type="text" name="question" placeholder="e.g. How do I manage cravings?" required>
                <button type="submit">Ask</button>
            </form>
            {% if answer %}
            <div class="response">
                <p><strong>Your Question:</strong><br>{{ request.form["question"] }}</p>
                <p><strong>Answer:</strong><br>{{ answer }}</p>
            </div>
            {% endif %}
        </div>
    </div>

    <div class="thought">
        <p>🧠 <strong>Thought of the Week:</strong><br>
        “Just for today, I will try to live through this day only, and not tackle all my problems at once.”</p>
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
            chunks, score = query_index(get_docs(), question)
            print(f"Top match score: {score:.2f}")
            if score > 0.3:
                answer = ask_gpt(question, chunks)
            else:
                answer = ask_general_gpt(question)
        except Exception as e:
            answer = f"⚠️ An error occurred: {str(e)}"
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
