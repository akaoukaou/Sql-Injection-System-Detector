import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from models.http_model import http_predict_query as http_predict, model_accuracy as http_score
from models.sql_model import extract_features_from_query , best_model_name , train_accuracy, val_accuracy, training_time,exported_feature_columns

app = Flask(__name__)

def format_sqlmodel_name(model_name):
    model_map = {
        "RandomForest": "Random Forest",
        "SVM": "SVM",
        "LogisticRegression": "Logistic Regression"
    }
    return model_map.get(model_name, model_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/http")
def http_page():
    return render_template("http.html", score=http_score)

@app.route("/sql")
def sql_page():
    return render_template(
        "sql.html",
        train_score=round(train_accuracy * 100, 2),
        val_score=round(val_accuracy * 100, 2),
        training_time=round(training_time, 2),
        best_model=format_sqlmodel_name(best_model_name) 
    )

@app.route("/analystic")
def analystic():
    return render_template("analystic.html")

@app.route("/files")
def files():
    return render_template("files.html")

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/predict-sql", methods=["POST"])
def predict_sql_route():
    data = request.get_json()
    query = data.get("query", "")
    selected_model = data.get("model", "randomforest").lower()

    model_path = f"saved_models/all_models/{selected_model}_model.pkl"

    # Charger le modèle choisi
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle : {e}")
        return jsonify({"error": f"Erreur de chargement du modèle : {e}"}), 500

    try:
        # 1. Extraire toutes les features (29)
        features_all = extract_features_from_query(query)
        # 2. Mapper les features avec leurs noms
        all_feature_names = [
            "LONGUEUR", "SCORE_INJECTION", "NB_KEYWORDS", "NB_SPECIAL_CHARS", "NB_QUOTES",
            "NB_COMMENT_SYNTAX", "RATIO_SCORE_LONGUEUR", "SCORE_COMPLEXITE", "CONTIENT_OR", "CONTIENT_QUOTE",
            "CONTIENT_COMMENT", "CONTIENT_UNION", "CONTIENT_EQUAL", "CONTIENT_PARENTHESES",
            "CONTIENT_TIME", "CONTIENT_FUNCTION", "CONTIENT_IN_CLAUSE", "CLASSE_LONGUEUR",
            "LONGUEUR_NORM", "SCORE_INJECTION_NORM", "NB_KEYWORDS_NORM", "NB_SPECIAL_CHAR_NORM",
            "NB_QUOTES_NORM", "NB_COMMENTS_NORM", "RATIO_SCORE_LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM",
            "CONTIENT_EXEC", "CONTIENT_SEMICOLON", "CONTIENT_UNION_SELECT"
        ]
        # 3. Liste des features attendues par le modèle
        selected_features = [
            "LONGUEUR",
            "SCORE_INJECTION",
            "RATIO_SCORE_LONGUEUR",
            "SCORE_COMPLEXITE",
            "LONGUEUR_NORM",
            "SCORE_INJECTION_NORM",
            "NB_KEYWORDS_NORM",
            "NB_SPECIAL_CHAR_NORM",
            "NB_QUOTES_NORM",
            "RATIO_SCORE_LONGUEUR_NORM",
            "SCORE_COMPLEXITE_NORM"
        ]
        # 4. Filtrer et ordonner les features à passer au modèle
        features_dict = dict(zip(all_feature_names, features_all))
        features_selected = [features_dict[name] for name in selected_features]

        # 5. Créer le DataFrame avec seulement les bonnes colonnes dans le bon ordre
        df = pd.DataFrame([features_selected], columns=selected_features)

        # 6. Faire la prédiction
        prediction = loaded_model.predict(df)[0]

        # Facultatif : calculer la confiance si le modèle le permet
        try:
            confidence = round(float(max(loaded_model.predict_proba(df)[0])), 3)
        except:
            confidence = None

        return jsonify({
            "result": int(prediction),
            "model": selected_model,
            "confidence": confidence
        })
    except Exception as e:
        print(f"❌ Erreur d’analyse : {e}")
        return jsonify({"error": f"Erreur d’analyse : {e}"}), 500



@app.route("/predict-http", methods=["POST"])
def predict_http_route():
    data = request.get_json()
    query = data.get("query", "")
    result = http_predict(query)
    return jsonify({
        "result": result,
        "type": "http",
        "confidence": http_score
    })

@app.route("/model-info")
def model_info():
    import json
    model_name = request.args.get("model", "").lower()
    stats_file = "saved_models/model_stats.json"
    
    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)
        
        if model_name not in stats:
            return jsonify({"error": "Modèle non trouvé"}), 404
        
        return jsonify(stats[model_name])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
