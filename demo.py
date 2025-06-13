import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from models.http_model import extract_features_from_http , best_model_name as http_best_model_name
from models.sql_model import extract_features_from_sql , best_model_name as sql_best_model_name

app = Flask(__name__)

def format_model_name(model_name):
    model_map = {
        "RandomForest": "Random Forest",
        "SVM": "SVM",
        "LogisticRegression": "Logistic Regression",
        "gradientboosting": "Gradient Boosting"
    }
    return model_map.get(model_name, model_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/http")
def http_page():
    return render_template("http.html", http_best_model=http_best_model_name)

@app.route("/sql")
def sql_page():
    return render_template("sql.html", sql_best_model=sql_best_model_name)

@app.route("/analystic")
def analystic():
    return render_template("analystic.html")

@app.route("/files")
def files():
    return render_template("files.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/model-info")
def model_info():
    model_name = request.args.get("model", "").lower()  # Nom du modèle (ex: 'randomforest')
    model_type = request.args.get("type", "")  # Type du modèle (ex: 'http' ou 'sql')

    if model_type == "http":
        stats_file = "saved_models/http_model_stats.json"
    elif model_type == "sql":
        stats_file = "saved_models/sql_model_stats.json"
    else:
        return jsonify({"error": "Modèle inconnu ou type incorrect"}), 400

    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)

        if model_name not in stats:
            return jsonify({"error": "Modèle non trouvé"}), 404

        return jsonify(stats[model_name])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-sql", methods=["POST"])
def predict_sql_route():
    data = request.get_json()
    query = data.get("query", "")
    selected_model = data.get("model", "randomforest").lower()

    model_path = f"saved_models/all_Sql_models/{selected_model}_model.pkl"

    # Charger le modèle choisi
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle : {e}")
        return jsonify({"error": f"Erreur de chargement du modèle : {e}"}), 500

    try:
        # 1. Extraire toutes les features
        features_all = extract_features_from_sql(query)
        # 2. Mapper les features avec leurs noms
        all_feature_names = [
            "LONGUEUR", "SCORE_INJECTION", "NB_KEYWORDS", "NB_SPECIAL_CHARS", "NB_QUOTES",
            "NB_COMMENT_SYNTAX", "RATIO_SCORE_LONGUEUR", "SCORE_COMPLEXITE", "CONTIENT_OR", "CONTIENT_QUOTE",
            "CONTIENT_COMMENT", "CONTIENT_UNION", "CONTIENT_EQUAL", "CONTIENT_PARENTHESES",
            "CONTIENT_TIME", "CONTIENT_FUNCTION", "CLASSE_LONGUEUR",
            "LONGUEUR_NORM", "SCORE_INJECTION_NORM", "NB_KEYWORDS_NORM", "NB_SPECIAL_CHAR_NORM",
            "NB_QUOTES_NORM", "NB_COMMENTS_NORM", "RATIO_SCORE_LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM"
        ]

        # 4. Filtrer et ordonner les features à passer au modèle
        features_dict = dict(zip(all_feature_names, features_all))

        features_selected = [features_dict[name] for name in all_feature_names]

        # 5. Créer le DataFrame avec seulement les bonnes colonnes dans le bon ordre
        df = pd.DataFrame([features_selected], columns=all_feature_names)

        # print("Features envoyées au modèle Flask :", features_selected)

        # 6. Faire la prédiction
        prediction = loaded_model.predict(df)[0]

        print("MODEL PATH UTILISÉ DANS FLASK :", model_path)
        print("Résultat brut du modèle :", prediction)

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
    selected_model = data.get("model", "randomforest").lower()

    model_path = f"saved_models/all_http_models/{selected_model}_model.pkl"

    print(f"MODEL PATH UTILISÉ DANS FLASK : {model_path}")

    # Charger le modèle choisi
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle : {e}")
        return jsonify({"error": f"Erreur de chargement du modèle : {e}"}), 500

    try:
        # Extraire les caractéristiques pour la requête HTTP
        features_all = extract_features_from_http(query)
        
        all_feature_names = [
            "CONTENT_LENGTH_NORM", "SPECIAL_CHAR_COUNT_NORM", "PARAM_COUNT_NORM", 
            "URL_SPECIAL_CHAR_COUNT_NORM", "URL_PARAM_COUNT_NORM", "URL_LENGTH_NORM",
            "D_CONTENT_SCORE_NORM", "URL_SCORE_NORM", "GLOBAL_SCORE_NORM",
            "NBKEYWORDSCONTENT_NORM", "NBKEYWORDSURL_NORM", "NBCOMMENTCONTENT_NORM", 
            "NBCOMMENTURL_NORM", "RATIOSCORELONGUEURCONTENT_NORM", "RATIOSCORELONGUEURURL_NORM", 
            "SCORECOMPLEXITECONTENT_NORM", "SCORECOMPLEXITEURL_NORM"
        ]
        
        features_dict = dict(zip(all_feature_names, features_all))
        features_selected = [features_dict[name] for name in all_feature_names]

        df = pd.DataFrame([features_selected], columns=all_feature_names)

        # print("Features envoyées au modèle Flask :", features_selected)

        # Faire la prédiction
        prediction = loaded_model.predict(df)[0]
        print(f"Résultat brut du modèle :", prediction)

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



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
