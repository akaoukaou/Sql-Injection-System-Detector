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

model_dir = "saved_models/all_Sql_models"
sql_stats_file = "saved_models/sql_model_stats.json"
MODELS = ["randomforest", "svm", "logisticregression"]

all_feature_names = [
    "LONGUEUR", "SCORE_INJECTION", "NB_KEYWORDS", "NB_SPECIAL_CHARS", "NB_QUOTES",
    "NB_COMMENT_SYNTAX", "RATIO_SCORE_LONGUEUR", "SCORE_COMPLEXITE", "CONTIENT_OR", "CONTIENT_QUOTE",
    "CONTIENT_COMMENT", "CONTIENT_UNION", "CONTIENT_EQUAL", "CONTIENT_PARENTHESES",
    "CONTIENT_TIME", "CONTIENT_FUNCTION", "CLASSE_LONGUEUR",
    "LONGUEUR_NORM", "SCORE_INJECTION_NORM", "NB_KEYWORDS_NORM", "NB_SPECIAL_CHAR_NORM",
    "NB_QUOTES_NORM", "NB_COMMENTS_NORM", "RATIO_SCORE_LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM"
]

models_already_trained = (
    os.path.exists(sql_stats_file)
    and all(os.path.exists(f"{model_dir}/{m}_model.pkl") for m in MODELS)
)

# 1. Charger et concaténer tous les fichiers
print("🔄 Chargement des datasets Vue SQL...")
try:
    df_train = pd.read_csv("dataset/SQL_Injec_NormTrain.csv")
    df_valid = pd.read_csv("dataset/SQL_Injec_Valid.csv")
    df_test = pd.read_csv("dataset/SQL_Injec_Test.csv")

    all_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    all_data_reduced = all_data[["LABEL"] + all_feature_names]

    # 2. Séparer 20% pour le test final (jamais utilisé en cross-validation)
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

# 3. Préparation des features
feature_columns = all_feature_names  # on s'assure de garder le bon ordre partout

X_trainval = df_trainval[feature_columns]
y_trainval = df_trainval['LABEL']

X_test = df_test_final[feature_columns]
y_test = df_test_final['LABEL']

print(f"\n📈 Distribution des classes (TrainVal):")
print(f"TrainVal - Benign: {sum(y_trainval == 0)}, Malicious: {sum(y_trainval == 1)} ({sum(y_trainval == 1)/len(y_trainval)*100:.1f}%)")
print(f"Test - Benign: {sum(y_test == 0)}, Malicious: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

print(f"\n📊 Statistiques des features (TrainVal):")
print(X_trainval.describe())

# 4. Définition des modèles
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

best_model = None
best_score = 0
best_model_name = ""
results = {}

#Entrainement et Stats sauvgarde
print("\n🔍 Début de la comparaison des modèles...")

# 1. Charger les stats existantes si le fichier existe
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
    # Si il manque soit le pkl soit l'info stat, on va le réentraîner
    if not (has_pkl and has_stat):
        models_to_train.append(model_key)

# 3. Réentraîner et sauvegarder les modèles manquants ou désynchronisés
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

    # Maj de la stat pour ce modèle
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

# 4. Charger tous les modèles + stats dans results (pour la sélection du best model)
results = {}
for model_name, model in models.items():
    model_key = model_name.lower()
    model_pkl = f"{model_dir}/{model_key}_model.pkl"
    # Charger le modèle existant et ses stats depuis le JSON
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

# Sauvegarder à la fin le JSON mis à jour
with open(sql_stats_file, "w") as f:
    json.dump(saved_stats, f, indent=4)

# Sélection du meilleur modèle (par val_accuracy)
best_model_name = max(results, key=lambda m: results[m]["val_accuracy"])
best_model = results[best_model_name]["model"]
best_score = results[best_model_name]["val_accuracy"]
train_accuracy = results[best_model_name]["train_accuracy"]
val_accuracy = results[best_model_name]["val_accuracy"]
training_time = results[best_model_name]["training_time"]

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"🎯 Score de validation croisée: {best_score:.4f}")

# Évaluation finale sur le test set
print(f"\n📊 Évaluation finale sur le test set...")
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Affichage clair des métriques
print(f"\n📋 Résultats complets sur l'ensemble de test :")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print(f"\n✅ RÉSULTATS FINAUX ({best_model_name}):")
print(f"  🎯 Train Score: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  🎯 Cross-Validation Score: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  🎯 Test Score: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  ⏱️ Training Time: {training_time:.2f} secondes")

print(f"\n📋 Rapport de classification (Test):")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malicious']))

print(f"\n📊 Matrice de confusion (Test):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Importance des features (si RandomForest)
if best_model_name == 'RandomForest':
    feature_importance = best_model.best_estimator_.named_steps['clf'].feature_importances_
    feature_names = feature_columns

    print(f"\n🔍 TOP 10 Features les plus importantes:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

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
    query = query.strip().lower()
    longueur = len(query)
    nb_quotes = query.count("'") + query.count('"')
    nb_special = len(re.findall(r'[^\w\s]', query))
    nb_keywords = len(re.findall(r'\b(select|insert|update|delete|from|where|union|drop|or|and|sleep|benchmark|char|ascii)\b', query))
    nb_comments = query.count("--") + query.count("/*")

    score_injection = sum([
        query.count(" or "),
        query.count("' or"),
        query.count('" or'),
        query.count("1=1"),
        query.count(" or 1=1"),
        query.count(" and "),
        len(re.findall(r"or\s+\d+=\d+", query)),
        len(re.findall(r"or\s+'.+'\s*=\s*'.+'", query)),
        query.count("union"),
        query.count("union select"),
        query.count("sleep("),
        query.count("benchmark"),
        query.count("char("),
        query.count("load_file"),
        query.count("information_schema"),
        query.count("--"),
        query.count("/*"),
        query.count(";"),
        query.count("exec"),
        query.count("xp_cmdshell"),
    ])

    ratio_score_longueur = score_injection / longueur if longueur > 0 else 0
    score_complexite = (nb_special + nb_keywords + nb_quotes) / longueur if longueur > 0 else 0

    contient_or = int(" or " in query)
    contient_quote = int("'" in query or '"' in query)
    contient_comment = int("--" in query or "/*" in query)
    contient_union = int("union" in query)
    contient_equal = int("=" in query)
    contient_parentheses = int("(" in query and ")" in query)
    contient_time = int("sleep" in query or "benchmark" in query)
    contient_function = int("concat" in query or "load_file" in query or "char(" in query)

    if longueur < 30:
        classe_longueur = 0
    elif longueur < 100:
        classe_longueur = 1
    else:
        classe_longueur = 2

    longueur_norm = min(longueur / 559, 1.0)
    score_injection_norm = min(score_injection / 12, 1.0)
    nb_keywords_norm = min(nb_keywords / 18, 1.0)
    nb_special_norm = min(nb_special / 80, 1.0)
    nb_quotes_norm = min(nb_quotes / 13, 1.0)
    nb_comments_norm = min(nb_comments / 4, 1.0)
    ratio_score_norm = min(ratio_score_longueur, 1.0)
    score_complexite_norm = min(score_complexite, 1.0)

    features = [
        longueur, score_injection, nb_keywords, nb_special, nb_quotes, nb_comments,
        ratio_score_longueur, score_complexite, contient_or, contient_quote,
        contient_comment, contient_union, contient_equal, contient_parentheses,
        contient_time, contient_function, classe_longueur,
        longueur_norm, score_injection_norm, nb_keywords_norm, nb_special_norm,
        nb_quotes_norm, nb_comments_norm, ratio_score_norm, score_complexite_norm
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

normal_queries = [
    "SELECT name, age FROM users WHERE id = 5;",
    "UPDATE products SET price = 19.99 WHERE product_id = 120;"
]
malicious_queries = [
    "SELECT * FROM users WHERE username = 'admin' --' AND password = '123';",
    "SELECT * FROM students WHERE name = '' OR 1=1 --';"
]
# --- Pour le debug et les tests en ligne de commande ---

print(f"\n🧪 Tests sur quelques échantillons du dataset:")
for i in range(min(5, len(X_test))):
    features = X_test.iloc[i].values
    true_label = y_test.iloc[i]
    pred_label = sql_predict(features)
    proba = sql_predict_proba(features)
    status = "✅" if pred_label == true_label else "❌"
    label_text = "MALICIOUS" if pred_label == 1 else "BENIGN"
    print(f"Test {i+1}: {status} Prédit: {label_text} (Vrai: {true_label}) - Confiance: {proba:.3f}")


def print_prediction(query):
    features_all = extract_features_from_sql(query)
    features_dict = dict(zip(all_feature_names, features_all))
    print(f"Features pour {query} :")
    for k in all_feature_names:
        print(f"{k}: {features_dict[k]}")
    features_selected = [features_dict[name] for name in all_feature_names]
    print("Features envoyées au modèle :", features_selected)
    pred = sql_predict(features_selected)
    label = "MALICIOUS" if pred == 1 else "BENIGN"
    print(f"Requête : {query}\n→ Prédiction du modèle : {label}\n")


print("=== Requêtes BENIGNES ===")
for q in normal_queries:
    print_prediction(q)

print("=== Requêtes MALICIEUSES ===")
for q in malicious_queries:
    print_prediction(q)
