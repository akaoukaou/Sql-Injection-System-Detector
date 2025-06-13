import pandas as pd
import time
import os
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import json
import re

# Define model directory and stats file
model_dir = "saved_models/all_http_models"
http_stats_file = "saved_models/http_model_stats.json"
#MODELS = ["randomforest", "svm", "gradientboosting"]
MODELS = ["randomforest", "gradientboosting"]


# Define feature names based on your dataset columns
all_feature_names = [
    "CONTENT_LENGTH_NORM", "SPECIAL_CHAR_COUNT_NORM", "PARAM_COUNT_NORM", 
    "URL_SPECIAL_CHAR_COUNT_NORM", "URL_PARAM_COUNT_NORM", "URL_LENGTH_NORM",
    "D_CONTENT_SCORE_NORM", "URL_SCORE_NORM", "GLOBAL_SCORE_NORM",
    "NBQUOTESCONTENT_NORM", "NBQUOTESURL_NORM", "NBKEYWORDSCONTENT_NORM", 
    "NBKEYWORDSURL_NORM", "NBCOMMENTCONTENT_NORM", "NBCOMMENTURL_NORM", 
    "RATIOSCORELONGUEURCONTENT_NORM", "RATIOSCORELONGUEURURL_NORM", 
    "SCORECOMPLEXITECONTENT_NORM", "SCORECOMPLEXITEURL_NORM"
]

# 1. Charger et concaténer tous les fichiers
print("🔄 Chargement des datasets HTTP...")

try:
    df_train = pd.read_csv("dataset/csic2010_train.csv")
    df_valid = pd.read_csv("dataset/csic2010_valid.csv")
    df_test = pd.read_csv("dataset/csic2010_test.csv")

    # Renommer les colonnes pour les aligner avec les noms attendus par le modèle
    df_train.rename(columns={'CLASSIFICATION': 'LABEL'}, inplace=True)
    df_valid.rename(columns={'CLASSIFICATION': 'LABEL'}, inplace=True)
    df_test.rename(columns={'CLASSIFICATION': 'LABEL'}, inplace=True)

    # Supprimer les colonnes inutiles (vides ou non pertinentes)
    df_train.drop(columns=['NBQUOTESCONTENT_NORM', 'NBQUOTESURL_NORM'], inplace=True)
    df_valid.drop(columns=['NBQUOTESCONTENT_NORM', 'NBQUOTESURL_NORM'], inplace=True)
    df_test.drop(columns=['NBQUOTESCONTENT_NORM', 'NBQUOTESURL_NORM'], inplace=True)

    # Mettre à jour la liste des caractéristiques en supprimant les colonnes
    all_feature_names_updated = [col for col in all_feature_names if col not in ['NBQUOTESCONTENT_NORM', 'NBQUOTESURL_NORM']]

    # Fusionner les datasets
    all_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    all_data_reduced = all_data[["LABEL"] + all_feature_names_updated]

    print("Colonnes du DataFrame après transformation :")
    # print(all_data_reduced.columns)

    # 2. Séparer 30% pour le test final (jamais utilisé en cross-validation)
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

# 3. Préparation des features
feature_columns = all_feature_names_updated  # Utilisez la nouvelle liste 

X_trainval = df_trainval[feature_columns]
y_trainval = df_trainval['LABEL']
X_test = df_test_final[feature_columns]
y_test = df_test_final['LABEL']

print(f"\n📈 Distribution des classes (TrainVal):")
print(f"TrainVal - Benign: {sum(y_trainval == 0)}, Malicious: {sum(y_trainval == 1)} ({sum(y_trainval == 1)/len(y_trainval)*100:.1f}%)")
print(f"Test - Benign: {sum(y_test == 0)}, Malicious: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

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
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
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
    'GradientBoosting': {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [3, 5, 7]
    }
}

best_model = None
best_score = 0
best_model_name = ""
results = {}

# Entraînement et sauvegarde des modèles
print("\n🔍 Début de la comparaison des modèles...")

# 1. Charger les stats existantes si le fichier existe
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

    # Mise à jour de la stat pour ce modèle
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

# Sauvegarder les statistiques mises à jour
with open(http_stats_file, "w") as f:
    json.dump(saved_stats, f, indent=4)

# 4. Sélectionner le meilleur modèle (par `val_accuracy`)
best_model_name = max(saved_stats, key=lambda m: saved_stats[m]["val_accuracy"])
best_model = pickle.load(open(f"{model_dir}/{best_model_name}_model.pkl", "rb"))
best_score = saved_stats[best_model_name]["val_accuracy"]

# Évaluation finale du meilleur modèle sur le jeu de test
print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"\n📋 Résultats complets sur l'ensemble de test :")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Affichage du rapport de classification et de la matrice de confusion
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
    url = request.split(' ')[1] if request.split(' ')[0] == "GET" else ''

    # Raw features
    content_length = len(request)
    special_char_count = len(re.findall(r'[^\w\s]', request))
    url_params = request.split('?')[-1] if '?' in request else ''
    param_count = len(url_params.split('&')) if url_params else 0
    url_special_char_count = len(re.findall(r'[^\w\s]', url))
    url_param_count = len(url_params.split('&')) if url_params else 0
    url_length = len(url)
    d_content_score = content_length / 100
    url_score = url_length / 100
    global_score = (d_content_score + url_score) / 2
    nb_keywords_content = len(re.findall(r'\b(select|insert|update|delete|drop|benchmark|exec)\b', request))
    nb_keywords_url = len(re.findall(r'\b(select|insert|update|delete|drop|benchmark|exec)\b', url))
    nb_comments_content = request.count("--") + request.count("/*")
    nb_comments_url = url.count("--") + url.count("/*")

    ratio_score_len_content = global_score / (len(request) + 1)
    ratio_score_len_url = url_score / (len(url) + 1)
    score_complexity_content = (special_char_count + nb_keywords_content) / (len(request) + 1)
    score_complexity_url = (url_special_char_count + nb_keywords_url) / (len(url) + 1)

    # Normalisation basique
    features = [
        content_length / 1000,                       # CONTENT_LENGTH_NORM
        special_char_count / 50,                     # SPECIAL_CHAR_COUNT_NORM
        param_count / 10,                            # PARAM_COUNT_NORM
        url_special_char_count / 20,                 # URL_SPECIAL_CHAR_COUNT_NORM
        url_param_count / 10,                        # URL_PARAM_COUNT_NORM
        url_length / 200,                            # URL_LENGTH_NORM
        d_content_score / 10,                        # D_CONTENT_SCORE_NORM
        url_score / 10,                              # URL_SCORE_NORM
        global_score / 10,                           # GLOBAL_SCORE_NORM
        nb_keywords_content / 10,                    # NBKEYWORDSCONTENT_NORM
        nb_keywords_url / 10,                        # NBKEYWORDSURL_NORM
        nb_comments_content / 5,                     # NBCOMMENTCONTENT_NORM
        nb_comments_url / 5,                         # NBCOMMENTURL_NORM
        ratio_score_len_content,                     # RATIOSCORELONGUEURCONTENT_NORM
        ratio_score_len_url,                         # RATIOSCORELONGUEURURL_NORM
        score_complexity_content,                    # SCORECOMPLEXITECONTENT_NORM
        score_complexity_url                         # SCORECOMPLEXITEURL_NORM
    ]

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

normal_requests = [
    "GET /index.html HTTP/1.1",
    "POST /login HTTP/1.1"
]

malicious_requests = [
    "GET /index.php?id=1 OR 1=1 -- HTTP/1.1",
    "GET /admin.php?username=admin&password=123456 HTTP/1.1"
]

# Prédiction sur les requêtes normales
for req in normal_requests:
    prediction = http_predict_from_query(req)
    print(f"Requête: {req}\nPrédiction: {'Benign' if prediction == 0 else 'Malicious'}")

# Prédiction sur les requêtes malveillantes
for req in malicious_requests:
    prediction = http_predict_from_query(req)
    print(f"Requête: {req}\nPrédiction: {'Benign' if prediction == 0 else 'Malicious'}")