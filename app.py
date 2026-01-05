from flask import Flask, request, render_template, session
from model_runner import generate_answer

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag_backend"))

app = Flask(__name__)
app.secret_key = "123"

cleared_once = False



@app.before_request
def clear_session_once():
    global cleared_once
    if not cleared_once:
        session.clear()
        cleared_once = True


@app.route("/", methods=["GET", "POST"])
def index():
    history = session.get("history", [])

    if request.method == "POST":
        q = request.form["question"]

        result = generate_answer(q, history)

        history.append(result)
        session["history"] = history

        return render_template("index.html", history=history)

    return render_template("index.html", history=history)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
