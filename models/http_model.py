import pandas as pd
import os
import re
import json
import time
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Directories & files
model_dir = "saved_models/all_http_models"
http_stats_file = "saved_models/http_model_stats.json"
#MODELS = ['svm', 'mlp', 'xgboost', 'lightgbm']
MODELS = ['mlp', 'xgboost', 'lightgbm']

# Features list
all_feature_names = [
    "METHODE_TYPE", "CONTIENT_EQUAL", "CONTIENT_QUOTE",
    "CONTIENT_COMMENT", "LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM",
    "NB_EQUALS_NORM", "RATIO_SCORE_LENGTH_NORM"
]
"""
    "METHODE_TYPE","CONTIENT_OR","CONTIENT_EQUAL","CONTIENT_QUOTE",
    "CONTIENT_COMMENT","CONTIENT_UNION","CONTIENT_SELECT","CONTIENT_FUNCTION",
    "LONGUEUR_NORM","SCORE_INJECTION_NORM","SCORE_COMPLEXITE_NORM",
    "NB_SQL_WORDS_NORM",# "NB_SPECIAL_CHARS_NORM","NB_QUOTES_NORM",
    "NB_EQUALS_NORM","RATIO_SCORE_LENGTH_NORM"
"""

# Charger et préparer les datasets
print("🔄 Chargement des datasets HTTP...")

try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_train = pd.read_csv("dataset/a_ids_train.csv")
    df_valid = pd.read_csv("dataset/a_ids_valid.csv")
    df_test = pd.read_csv("dataset/a_ids_test.csv")

    all_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    all_data_reduced = all_data[["LABEL"] + all_feature_names]

    df_trainval, df_test_final = train_test_split(
        all_data_reduced, test_size=0.30, stratify=all_data_reduced['LABEL'], random_state=42
    )

    print(f"✅ Jeux fusionnés : total {len(all_data)} échantillons")
    print(f"  ⮕ Entraînement + validation croisée : {len(df_trainval)}")
    print(f"  ⮕ Test final : {len(df_test_final)}")

except FileNotFoundError as e:
    print(f"❌ Erreur: {e}")
    print("Vérifiez que les fichiers CSV sont dans le bon répertoire")
    exit(1)

feature_columns = all_feature_names
X_trainval = df_trainval[feature_columns]
y_trainval = df_trainval['LABEL']
X_test = df_test_final[feature_columns]
y_test = df_test_final['LABEL']

print(f"\n📈 Distribution des classes (TrainVal):")
print(f"TrainVal - Benign: {sum(y_trainval == 0)}, Malicious: {sum(y_trainval == 1)} ({sum(y_trainval == 1)/len(y_trainval)*100:.1f}%)")
print(f"Test - Benign: {sum(y_test == 0)}, Malicious: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

# Définition des modèles et hyperparamètres
models = {
    # 'SVM': Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('clf', SVC(probability=True, random_state=42))
    # ]),
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            random_state=42,
            max_iter=300,
            early_stopping=True,
            alpha=0.0005,
            learning_rate_init=0.001,
            validation_fraction=0.15,
            hidden_layer_sizes=(32, 16),
            batch_size=128,
            activation='relu',
            solver='adam'
        ))
    ]),
    'XGBoost': Pipeline([
        ('clf', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            enable_categorical=False,
            scale_pos_weight=3,  # Pondere les cas positifs
            max_depth=3,  # Limite la profondeur
            learning_rate=0.05,  # Taux plus bas
            subsample=0.8,  # Stochastic sampling
            colsample_bytree=0.8,
            reg_alpha=1,  # L1 regularization
            reg_lambda=1  # L2 regularization
        ))
    ]),
    'LightGBM': Pipeline([
        #('scaler', StandardScaler()),
        ('clf', LGBMClassifier(verbose=-1, random_state=42))
    ])
}

params = {
    #'SVM': {'clf__C': [1, 10],'clf__kernel': ['rbf', 'linear']},
    'MLP': {
        'clf__hidden_layer_sizes': [(32, 16), (64, 32)],
        'clf__alpha': [0.0001, 0.001, 0.005],
        'clf__learning_rate_init': [0.001, 0.0005],
        'clf__activation': ['relu'],
        'clf__batch_size': [128, 256],
        'clf__early_stopping': [True],
        'clf__max_iter': [200]
    },
    'XGBoost': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.01, 0.05],
        'clf__subsample': [0.6, 0.8],
        'clf__colsample_bytree': [0.6, 0.8],
        'clf__reg_alpha': [0, 1],
        'clf__reg_lambda': [0, 1]
    },
    'LightGBM': {
        'clf__n_estimators': [100],
        'clf__learning_rate': [0.1],
        'clf__max_depth': [3, 5]
    }
}

# ---------------------------------------------------------
# *** GESTION DES STATS ***
# ---------------------------------------------------------

print("\n🔍 Début de la comparaison des modèles...")

# 1. Charger stats existantes
if os.path.exists(http_stats_file):
    with open(http_stats_file, "r") as f:
        saved_stats = json.load(f)
else:
    saved_stats = {}

# 2. Identifier les modèles à réentraîner
models_to_train = []
for model_key in MODELS:
    model_pkl = f"{model_dir}/{model_key}_model.pkl"
    has_pkl = os.path.exists(model_pkl)
    has_stat = model_key in saved_stats
    if not (has_pkl and has_stat):
        models_to_train.append(model_key)

# 3. Réentraîner et MAJ stats en mémoire (RAM)
for model_key in models_to_train:
    model_name = [k for k in models.keys() if k.lower() == model_key][0]
    model = models[model_name]
    print(f"\n🔧 (Re)Entraînement du modèle: {model_name}")
    grid_search = GridSearchCV(
        model, 
        param_grid=params[model_name], 
        cv=5, 
        scoring='f1',  # Optimiser pour F1-score
        n_jobs=-1, 
        verbose=1
    )
    start_time = time.time()
    grid_search.fit(X_trainval, y_trainval)

    model_training_time = time.time() - start_time
    train_acc = grid_search.score(X_trainval, y_trainval)
    val_acc = grid_search.best_score_
    y_pred_test = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    # Sauvegarde du modèle entraîné
    os.makedirs(model_dir, exist_ok=True)
    model_pkl = f"{model_dir}/{model_key}_model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(grid_search, f)
    print(f"💾 Modèle {model_name} sauvegardé dans {model_pkl}")

    # Mise à jour des stats (RAM, pas encore écrit sur disque)
    saved_stats[model_key] = {
        "train_accuracy": round(train_acc * 100, 2),
        "val_accuracy": round(val_acc * 100, 2),
        "test_accuracy": round(test_accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "cv_score": round(val_acc * 100, 2),
        "training_time": round(model_training_time, 2),
        "best_params": grid_search.best_params_
    }

# 4. Sauvegarde unique du JSON
with open(http_stats_file, "w") as f:
    json.dump(saved_stats, f, indent=4)

# 5. Rechargement de tous les modèles + stats pour sélection du meilleur
results = {}
for model_name, model in models.items():
    model_key = model_name.lower()
    model_pkl = f"{model_dir}/{model_key}_model.pkl"
    with open(model_pkl, "rb") as f:
        loaded_model = pickle.load(f)
    info = saved_stats.get(model_key, {})
    results[model_name] = {
        "model": loaded_model,
        "train_accuracy": info.get("train_accuracy", 0) / 100,
        "val_accuracy": info.get("val_accuracy", 0) / 100,
        "test_accuracy": info.get("test_accuracy", 0) / 100,
        "precision": info.get("precision", 0) / 100,
        "recall": info.get("recall", 0) / 100,
        "f1_score": info.get("f1_score", 0) / 100,
        "training_time": info.get("training_time", 0),
        "best_params": info.get("best_params", {}),
        "cv_score": info.get("cv_score", 0) / 100,
    }

# 6. Sélection du meilleur modèle
best_model_name = max(results, key=lambda m: results[m]["val_accuracy"])
best_model = results[best_model_name]["model"]
best_score = results[best_model_name]["val_accuracy"]
train_accuracy = results[best_model_name]["train_accuracy"]
val_accuracy = results[best_model_name]["val_accuracy"]
training_time = results[best_model_name]["training_time"]

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"🎯 Score de validation croisée: {best_score:.4f}")

# 7. Évaluation finale sur le test set
y_test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"\n📋 Résultats complets sur l'ensemble de test :")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print(f"\n✅ RÉSULTATS FINAUX ({best_model_name}):")
print(f"  🎯 Train Score: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  🎯 Cross-Validation Score: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  🎯 Test Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ⏱️ Training Time: {training_time:.2f} secondes")

print(f"\n📋 Rapport de classification (Test):")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malicious']))

print(f"\n📊 Matrice de confusion (Test):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Sauvegarde du meilleur modèle
try:
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/best_http_model_{best_model_name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n💾 Meilleur modèle sauvegardé: {model_path}")
except Exception as e:
    print(f"⚠️ Impossible de sauvegarder le modèle: {e}")



# ---------------------------
# Fonctions d’extraction/features + prédiction
# ---------------------------

def extract_features_from_http(request):
    """
    features = {
        "METHODE_TYPE": 0.0,#"CONTIENT_OR": 0.0,"CONTIENT_EQUAL": 0.0,
        "CONTIENT_QUOTE": 0.0,"CONTIENT_COMMENT": 0.0,#"CONTIENT_UNION": 0.0,
        #"CONTIENT_SELECT": 0.0,#"CONTIENT_FUNCTION": 0.0,"LONGUEUR_NORM": 0.0,
        "SCORE_INJECTION_NORM": 0.0,"SCORE_COMPLEXITE_NORM": 0.0,#"NB_SQL_WORDS_NORM": 0.0,
        "NB_EQUALS_NORM": 0.0,"RATIO_SCORE_LENGTH_NORM": 0.0
    }
    """

    features = {
        "METHODE_TYPE": 0.0,
        "CONTIENT_EQUAL": 0.0,
        "CONTIENT_QUOTE": 0.0,
        "CONTIENT_COMMENT": 0.0,
        "LONGUEUR_NORM": 0.0,
        "SCORE_INJECTION_NORM": 0.0,
        "SCORE_COMPLEXITE_NORM": 0.0,
        "NB_EQUALS_NORM": 0.0,
        "RATIO_SCORE_LENGTH_NORM": 0.0
    }

    try:
        # 1. Séparer headers et body
        parts = request.split('\n\n', 1)
        headers = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        first_line = headers.split('\n')[0] if headers else ""

        # 2. METHODE_TYPE : POST=1, autres=0
        method_post = first_line.strip().upper().startswith('POST')
        features["METHODE_TYPE"] = 1.0 if method_post else 0.0

        # 3. Extraire URL
        url = re.sub(r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|CONNECT)\s+|\s+HTTP/.*', 
                     '', first_line.strip(), flags=re.IGNORECASE)
        parsed_url = urlparse(url)
        query = parsed_url.query

        # 4. Contenu pour analyse : corps + query string
        content = body + " " + query

        # 5. Normalisation longueur (max 1000 chars)
        features["LONGUEUR_NORM"] = min(len(content) / 1000, 1.0)

        # 6. Recherche mots clés SQL
        features["CONTIENT_OR"] = min(len(re.findall(r'\bOR\b', content, re.IGNORECASE)) / 5.0, 1.0)
        features["CONTIENT_EQUAL"] = min(len(re.findall(r'=', content)) / 10.0, 1.0)
        features["CONTIENT_QUOTE"] = min(len(re.findall(r'[\'"]', content)) / 5.0, 1.0)
        features["CONTIENT_COMMENT"] = min(len(re.findall(r'(--|\#|/\*|\*/)', content)) / 3.0, 1.0)
        features["CONTIENT_UNION"] = min(len(re.findall(r'\bUNION\b', content, re.IGNORECASE)) / 2.0, 1.0)
        features["CONTIENT_SELECT"] = min(len(re.findall(r'\bSELECT\b', content, re.IGNORECASE)) / 2.0, 1.0)
        features["CONTIENT_FUNCTION"] = min(len(re.findall(r'\b(EXEC|DECLARE|CALL|EVAL|CAST)\b', content, re.IGNORECASE)) / 3.0, 1.0)

        # 7. Nombre de mots SQL
        sql_keywords = r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b'
        features["NB_SQL_WORDS_NORM"] = min(len(re.findall(sql_keywords, content, re.IGNORECASE)) / 5.0, 1.0)

        # 8. Score injection : compter motifs typiques
        injection_patterns = [
            r'\bUNION\s+SELECT\b', r'\bDROP\s+TABLE\b', r'\bINSERT\s+INTO\b', r'\bDELETE\s+FROM\b',
            r'\bEXEC\s*\(', r'WAITFOR\s+DELAY', r'<\s*SCRIPT\b', r'1\s*=\s*1', r'\bSLEEP\s*\(',
            r'\bXP_', r'--\s', r';.*;', r'/\*.*\*/', r'<\?php', r'javascript:', r'\.\./',
            r'\bOR\b.*\b1\s*=\s*1\b', r"'\s*--", r';.*--', r'<\s*script\b', r'\bWAITFOR\s+DELAY\b',
            r'\bSLEEP\s*\(', r'\bBENCHMARK\s*\(', r'\bXP_', r'\bEXEC\b', r'\bUNION\b.*\bSELECT\b',
            r'\bOR\b\s+\d+\s*=\s*\d+', r"'\s*(--|#|/\*|\*/)", r';.*--', r'<\s*script[^>]*>.*<\s*/\s*script\s*>',
            r'\.\./(\.\./)*', r'union\s+select', r'exec(\s|\()+', r'waitfor\s+delay',
            r'benchmark\s*\(', r'load_file\s*\(', r'into\s+(outfile|dumpfile)', r'xpath\s*\(', r'xmltype\s*\(',
        ]
        injection_score = sum(1 if re.search(p, content, re.IGNORECASE) else 0 for p in injection_patterns)
        features["SCORE_INJECTION_NORM"] = min(injection_score / 3.0, 1.0)

        # 9. Score complexité
        complexity_factors = (
            len(re.findall(r'[^\w\s]', content)) + 
            len(re.findall(r'\b(AND|OR|NOT)\b', content, re.IGNORECASE)) + 
            len(re.findall(r'\(.*?\)', content))
        )
        features["SCORE_COMPLEXITE_NORM"] = min(complexity_factors / 30.0, 1.0)

        # 10. Ratio score / longueur
        if len(content) > 0:
            features["RATIO_SCORE_LENGTH_NORM"] = min(
                (features["SCORE_INJECTION_NORM"] + features["SCORE_COMPLEXITE_NORM"]) /
                (len(content) / 1000.0 + 0.001),
                1.0
            )

        seuil_injection = 0.3
        seuil_complexite = 0.3

        benign_post_patterns = ['username=', 'password=', 'name=', 'age=', 'email=']

        if features["METHODE_TYPE"] == 1.0:
            content_lower = content.lower()
            if any(p in content_lower for p in benign_post_patterns):
                if (features["SCORE_INJECTION_NORM"] < seuil_injection + 0.2 and
                    features["SCORE_COMPLEXITE_NORM"] < seuil_complexite + 0.2):
                    features["METHODE_TYPE"] = 0.0
            else:
                if (features["SCORE_INJECTION_NORM"] < seuil_injection and
                    features["SCORE_COMPLEXITE_NORM"] < seuil_complexite):
                    features["METHODE_TYPE"] = 0.0

    
    except Exception as e:
        print(f"⚠️ Feature extraction error: {str(e)}")
    
    #print("✅ Features générées :", features.keys())
    return features

# ---------------------------

def http_predict(query_features):
    # Vérifier les features sont sous forme de liste ou tableau avec la bonne longueur
    if isinstance(query_features, dict):
        features_selected = [query_features[name] for name in all_feature_names]
    elif isinstance(query_features, list) or isinstance(query_features, np.ndarray):
        if len(query_features) == len(all_feature_names):
            features_selected = query_features
        else:
            print(f"❌ Erreur: Attendu {len(all_feature_names)} features, reçu {len(query_features)}")
            return 0
    else:
        print("⚠️ Attention: Ce modèle nécessite des features extraites, pas une requête brute")
        return 0

    # Créer le DataFrame pour passer les features au modèle
    df = pd.DataFrame([features_selected], columns=all_feature_names)

    # Faire la prédiction
    return int(best_model.predict(df)[0])

# ---------------------------

def http_predict_proba(query_features):
    if isinstance(query_features, dict):
        features_selected = [query_features[name] for name in all_feature_names]
    elif isinstance(query_features, list) or isinstance(query_features, np.ndarray):
        if len(query_features) == len(all_feature_names):
            features_selected = query_features
        else:
            return 0.5  # Valeur par défaut en cas d'erreur
    else:
        return 0.5  # Valeur par défaut en cas d'erreur

    df = pd.DataFrame([features_selected], columns=all_feature_names)
    return max(best_model.predict_proba(df)[0])  # Retourne la probabilité maximale

# ---------------------------

def http_predict_from_query(query):
    features_all = extract_features_from_http(query)

    return http_predict(features_all)

# ---------------------------
# Exemple de requêtes HTTP à tester
# ---------------------------

test_queries = [
    # Requêtes normales (benignes)
    ("GET /index.html HTTP/1.1", 0),
    ("POST /login HTTP/1.1\nContent-Length: 20\n\nusername=test&pwd=123", 0),
    ("GET /products/list?page=3&order=asc HTTP/1.1", 0),
    ("GET /profile/view?user=alice HTTP/1.1", 0),
    ("POST /api/v1/update HTTP/1.1\nContent-Length: 33\n\nname=Tom&age=32&country=france", 0),

    # Attaques classiques
    ("GET /index.php?id=1 OR 1=1 -- HTTP/1.1", 1),
    ("POST /login.php HTTP/1.1\nContent-Length: 41\n\nusername=admin' --&password=irrelevant", 1),
    ("GET /search.php?q=UNION+SELECT+1,2,3 HTTP/1.1", 1),
    ("GET /products.php?id=5; DROP TABLE users; -- HTTP/1.1", 1),
    ("GET /admin.php?username=admin&password=123456 HTTP/1.1", 1),
    ("GET /home.php?query=<script>alert(1)</script> HTTP/1.1", 1),

    # Attaques subtiles/furtives
    ("GET /data.php?item=foo'/**/OR/**/1=1-- HTTP/1.1", 1),
    ("POST /api/upload HTTP/1.1\nContent-Length: 49\n\nfile=../../../../etc/passwd&name=x", 1),
    ("GET /page.php?id=2;WAITFOR DELAY '0:0:5'-- HTTP/1.1", 1),

    # Faux positifs potentiels (normaux mais un peu “louches”)
    ("GET /admin.js?debug=true HTTP/1.1", 0),
    ("GET /backup.tar.gz HTTP/1.1", 0),
    ("GET /contact.php?subject=orchestra HTTP/1.1", 0),
    ("POST /api/save HTTP/1.1\nContent-Length: 18\n\nnote=DROP+by+soon", 0),
    ("GET /newsletter.php?email=foo@bar.com HTTP/1.1", 0),
]

print("\n===== HTTP TEST =====")
for q, expected in test_queries:
    pred = http_predict_from_query(q)
    verdict = "✅" if pred == expected else "❌"
    label_txt = "🟡 MALICIOUS" if pred == 1 else "⚪ BENIGN"
    print(f"{verdict} Requête : {q[:70]}... → Prédit :  {label_txt} (Attendu : {'🟡 MALICIOUS' if expected==1 else '⚪ BENIGN'})")


# Requête bénigne avec paramètres (devrait retourner 0)
print(http_predict_from_query("GET /search?q=hello&page=1 HTTP/1.1"))

# Requête malveillante (devrait retourner 1)
print(http_predict_from_query("GET /index.php?id=1 UNION SELECT * FROM users-- HTTP/1.1"))
