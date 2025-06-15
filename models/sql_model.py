import pandas as pd
import time
import os
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import re
import json

# Directories & files
model_dir = "saved_models/all_Sql_models"
sql_stats_file = "saved_models/sql_model_stats.json"
MODELS = ["randomforest" ,"logisticregression"]

# Feature names
all_feature_names = [
    "LONGUEUR", "SCORE_INJECTION", "NB_KEYWORDS", "NB_SPECIAL_CHARS", "NB_QUOTES",
    "NB_COMMENT_SYNTAX", "RATIO_SCORE_LONGUEUR", "SCORE_COMPLEXITE", "CONTIENT_OR", "CONTIENT_QUOTE",
    "CONTIENT_COMMENT", "CONTIENT_UNION", "CONTIENT_EQUAL", "CONTIENT_PARENTHESES",
    "CONTIENT_TIME", "CONTIENT_FUNCTION", "CLASSE_LONGUEUR",
    "LONGUEUR_NORM", "SCORE_INJECTION_NORM", "NB_KEYWORDS_NORM", "NB_SPECIAL_CHAR_NORM",
    "NB_QUOTES_NORM", "NB_COMMENTS_NORM", "RATIO_SCORE_LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM"
]

# Charger et préparer les datasets
print("🔄 Chargement des datasets Vue SQL...")

try:
    df_train = pd.read_csv("dataset/SQL_Injec_NormTrain.csv")
    df_valid = pd.read_csv("dataset/SQL_Injec_Valid.csv")
    df_test = pd.read_csv("dataset/SQL_Injec_Test.csv")

    all_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    all_data_reduced = all_data[["LABEL"] + all_feature_names]

    df_trainval, df_test_final = train_test_split(
        all_data_reduced, test_size=0.20, stratify=all_data_reduced['LABEL'], random_state=42
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
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42, probability=True))
    ]),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
}

params = {
    'RandomForest': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5]
    },
    'SVM': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['rbf', 'linear'],
        'clf__gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear', 'saga']
    }
}

# ---------------------------------------------------------
# *** GESTION DES STATS ***
# ---------------------------------------------------------

print("\n🔍 Début de la comparaison des modèles...")

# 1. Charger stats existantes
if os.path.exists(sql_stats_file):
    with open(sql_stats_file, "r") as f:
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

# 3. Entraîner et MAJ stats (en RAM)
for model_key in models_to_train:
    model_name = [k for k in models.keys() if k.lower() == model_key][0]
    model = models[model_name]
    print(f"\n🔧 (Re)Entraînement du modèle: {model_name}")
    grid_search = GridSearchCV(model, param_grid=params[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
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

    # MAJ stats (en RAM)
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

# 4. Sauvegarde unique des stats
with open(sql_stats_file, "w") as f:
    json.dump(saved_stats, f, indent=4)

# 5. Recharger tous les modèles + stats pour sélection du meilleur
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
    model_path = f'saved_models/best_sql_model_{best_model_name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n💾 Meilleur modèle sauvegardé: {model_path}")
except Exception as e:
    print(f"⚠️ Impossible de sauvegarder le modèle: {e}")



# ---------------------------
# Fonctions d’extraction/features + prédiction
# ---------------------------

def extract_features_from_sql(query):
    q = query.strip().lower()
    longueur = len(q)

    # 1. LONGUEUR
    longueur = len(q)
    
    # 2. SCORE_INJECTION : nombre de patterns d’injection simples repérés (pour donner de la “matière” au ML, pas pour juger)
    patterns_injection = [
        r"or\s+\d+=\d+", r"union\s+select", r"sleep\s*\(", r"benchmark\s*\(", r"waitfor\s+delay", 
        r"information_schema", r"admin'\s*--", r"drop\s+table", r"delete\s+from", r"insert\s+into", r"--", r"#", r"/\*.*?\*/"
    ]
    score_injection = sum(len(re.findall(p, q)) for p in patterns_injection)
    
    # 3. NB_KEYWORDS : nombre total de mots-clés SQL (sans pondération)
    keywords_sql = ["select", "from", "where", "and", "or", "insert", "update", "delete", "drop",
                    "union", "exec", "sleep", "benchmark", "like", "count", "in", "set", "values"]
    nb_keywords = sum(len(re.findall(r"\b" + k + r"\b", q)) for k in keywords_sql)

    # 4. NB_SPECIAL_CHARS : nombre de caractères spéciaux
    special_chars = ["'", '"', "`", ";", "(", ")", "-", "#", "/", "*", "%", "=", "<", ">"]
    nb_special_chars = sum(q.count(s) for s in special_chars)

    # 5. NB_QUOTES : nombre total de quotes
    nb_quotes = q.count("'") + q.count('"') + q.count("`")

    # 6. NB_COMMENT_SYNTAX : nombre de syntaxes de commentaires
    nb_comment_syntax = len(re.findall(r"(--|#|/\*.*?\*/)", q))

    # 7. RATIO_SCORE_LONGUEUR : rapport score injection / longueur
    ratio_score_longueur = score_injection / longueur if longueur > 0 else 0

    # 8. SCORE_COMPLEXITE : nombre de mots-clés avancés
    keywords_complex = ["join", "group", "order", "having", "distinct", "case", "when",
                       "then", "else", "end", "sum", "avg", "limit", "substring"]
    score_complexite = sum(len(re.findall(r"\b" + k + r"\b", q)) for k in keywords_complex)

    # 9. CONTIENT_OR : 1 si "or" existe dans la requête, 0 sinon
    contient_or = int(" or " in q)

    # 10. CONTIENT_QUOTE : 1 si au moins une quote (simple, double ou backtick)
    contient_quote = int(nb_quotes > 0)

    # 11. CONTIENT_COMMENT : 1 si présence de --, # ou /* ... */
    contient_comment = int(nb_comment_syntax > 0)

    # 12. CONTIENT_UNION : 1 si "union" dans la requête
    contient_union = int("union" in q)

    # 13. CONTIENT_EQUAL : 1 si "=" dans la requête
    contient_equal = int("=" in q)

    # 14. CONTIENT_PARENTHESES : 1 si ( et )
    contient_parentheses = int(("(" in q) and (")" in q))

    # 15. CONTIENT_TIME : 1 si "sleep", "benchmark" ou "waitfor" dans la requête
    contient_time = int(any(w in q for w in ["sleep", "benchmark", "waitfor"]))

    # 16. CONTIENT_FUNCTION : 1 si présence de mots-clés de fonctions SQL suspects
    functions = ["load_file", "benchmark", "sleep", "exec", "execute", "information_schema", "sys."]
    contient_function = int(any(f in q for f in functions))

    # 17. CLASSE_LONGUEUR : 0 = court, 1 = moyen, 2 = long
    if longueur < 50:
        classe_longueur = 0
    elif longueur < 150:
        classe_longueur = 1
    else:
        classe_longueur = 2

    # --- Features normalisées ---
    LONG_MAX = 2000
    INJECTION_MAX = 10
    KEYWORD_MAX = 20
    SPECIAL_MAX = 20
    QUOTE_MAX = 10
    COMMENT_MAX = 5
    RATIO_MAX = 0.2
    COMPLEX_MAX = 10

    longueur_norm = min(longueur / LONG_MAX, 1.0)
    score_injection_norm = min(score_injection / INJECTION_MAX, 1.0)
    nb_keywords_norm = min(nb_keywords / KEYWORD_MAX, 1.0)
    nb_special_char_norm = min(nb_special_chars / SPECIAL_MAX, 1.0)
    nb_quotes_norm = min(nb_quotes / QUOTE_MAX, 1.0)
    nb_comments_norm = min(nb_comment_syntax / COMMENT_MAX, 1.0)
    ratio_score_norm = min(ratio_score_longueur / RATIO_MAX, 1.0)
    score_complexite_norm = min(score_complexite / COMPLEX_MAX, 1.0)

    # --- Dictionnaire ---
    features = [
        longueur,
        score_injection,
        nb_keywords,
        nb_special_chars,
        nb_quotes,
        nb_comment_syntax,
        ratio_score_longueur,
        score_complexite,
        contient_or,
        contient_quote,
        contient_comment,
        contient_union,
        contient_equal,
        contient_parentheses,
        contient_time,
        contient_function,
        classe_longueur,
        longueur_norm,
        score_injection_norm,
        nb_keywords_norm,
        nb_special_char_norm,
        nb_quotes_norm,
        nb_comments_norm,
        ratio_score_norm,
        score_complexite_norm
    ]
    return features

# ---------------------------

def sql_predict(query_features):
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

    df = pd.DataFrame([features_selected], columns=all_feature_names)
    return int(best_model.predict(df)[0])

# ---------------------------

def sql_predict_proba(query_features):
    if isinstance(query_features, dict):
        features_selected = [query_features[name] for name in all_feature_names]
    elif isinstance(query_features, list) or isinstance(query_features, np.ndarray):
        if len(query_features) == len(all_feature_names):
            features_selected = query_features
        else:
            return 0.5
    else:
        return 0.5
    df = pd.DataFrame([features_selected], columns=all_feature_names)
    return max(best_model.predict_proba(df)[0])

# ---------------------------

def sql_predict_from_query(query):
    features_all = extract_features_from_sql(query)
    return sql_predict(features_all)


# ---------------------------
# Exemple de requêtes SQL à tester
# ---------------------------

test_queries = [
    # Normales
    ("SELECT * FROM users WHERE email = 'user@example.com';", 0),
    ("UPDATE orders SET status = 'confirmed' WHERE order_id = 444;", 0),
    ("DELETE FROM sessions WHERE user_id = 42;", 0),
    ("INSERT INTO logs (date, action) VALUES ('2023-06-13', 'login');", 0),
    ("SELECT COUNT(*) FROM transactions WHERE status = 'ok';", 0),
    ("SELECT * FROM orders WHERE order_id IN (1, 2, 3, 4);", 0),
    ("UPDATE users SET password = 'abc123' WHERE id = 100;", 0),
    ("SELECT * FROM clients WHERE name LIKE '%smith%';", 0),
    # Attaques
    ("SELECT * FROM users WHERE username = '' OR 1=1 -- ;", 1),
    ("SELECT * FROM clients WHERE name = 'admin' # OR '1'='1';", 1),
    ("SELECT * FROM orders WHERE id = 5; DROP TABLE orders;", 1),
    ("SELECT * FROM users WHERE id = 1 UNION SELECT credit_card FROM cards;", 1),
    ("SELECT * FROM logins WHERE username = 'test' OR 'a'='a';", 1),
    ("SELECT * FROM products WHERE name = '' OR SLEEP(5) -- ;", 1),
    # Attaques subtiles
    ("SELECT * FROM users WHERE username = 'admin'/*' or '1'='1*/';", 1),
    ("SELECT * FROM items WHERE description = '' UNION SELECT version()--';", 1),
    ("SELECT * FROM inventory WHERE name = 'item' OR ascii(substring((SELECT password FROM users LIMIT 1),1,1))=97 --';", 1),
    ("SELECT * FROM logs WHERE action = 'delete' OR benchmark(1000000,MD5(1))--';", 1),
    # Faux positifs potentiels
    ("SELECT * FROM orders WHERE id = 10 AND (status = 'ok' OR status = 'pending');", 0),
    ("SELECT * FROM accounts WHERE (name = 'alice' AND type = 'premium');", 0),
    ("SELECT * FROM catalogue WHERE (category = 'books' AND price <= 20) OR (category = 'cds' AND price <= 10);", 0),
    ("SELECT * FROM payments WHERE details = 'VISA (**** **** **** 1234)';", 0),
]

print("\n===== TEST =====")
for q, expected in test_queries:
    pred = sql_predict_from_query(q)
    verdict = "✅" if pred == expected else "❌"
    label_txt = "🟡 MALICIOUS" if pred == 1 else "⚪ BENIGN"
    print(f"{verdict} Requête : {q[:70]}... → Prédit :  {label_txt} (Attendu : {'🟡 MALICIOUS' if expected==1 else '⚪ BENIGN'})")
