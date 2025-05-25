from flask import Flask, render_template, request, jsonify
from models.http_model import predict_query as predict_http, model_accuracy as http_score
from models.sql_model import predict_query as predict_sql, model_accuracy as sql_score

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/http")
def http_page():
    return render_template("http.html", score=http_score)

@app.route("/sql")
def sql_page():
    return render_template("sql.html", score=sql_score)

@app.route("/analystic")
def analystic():
    return render_template("analystic.html")

@app.route("/files")
def files():
    return render_template("files.html")

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    model_type = data.get("type", "http")

    if model_type == "sql":
        result = predict_sql(query)
    else:
        result = predict_http(query)

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
