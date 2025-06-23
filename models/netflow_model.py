import os
import re
import time
import math
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Directories & files
model_dir = "saved_models/all_net_models"
net_stats_file = "saved_models/net_model_stats.json"
MODELS = ["randomforest", "xgboost", "gradientboosting"]

# Feature names
all_feature_names = [
    "BYTE_RATE_LOGNORM", "PKT_RATE_LOGNORM", "DURATION_NORM", "AVG_PACKET_SIZE_NORM"
]

# Charger et préparer les datasets
print("🔄 Chargement des datasets Vue NET...")

try:
    df_train = pd.read_csv("dataset/NET_attack_train.csv")
    df_valid = pd.read_csv("dataset/NET_attack_valid.csv")
    df_test = pd.read_csv("dataset/NET_attack_test.csv")

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
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
}

params = {
    'RandomForest': {
        'clf__n_estimators': [100, 200],  # Augmenter le nombre d'arbres
        'clf__max_depth': [10, 20],        # Profondeur des arbres
        'clf__min_samples_split': [5, 10]  # Diviser les échantillons moins fréquemment pour éviter l'overfitting
    },
    'XGBoost': {
        'clf__n_estimators': [200, 300],  # Augmenter les arbres
        'clf__max_depth': [6, 10],         # Augmenter la profondeur pour capter des patterns complexes
        'clf__learning_rate': [0.01, 0.1]  # Réduire le taux d'apprentissage
    },
    'GradientBoosting': {
        'clf__n_estimators': [200, 300],   # Plus d'arbres pour capturer plus de variations
        'clf__learning_rate': [0.01, 0.05], # Réduire le taux d'apprentissage pour plus de précision
        'clf__max_depth': [6, 10]           # Plus de profondeur pour détecter des relations plus complexes
    }
}


# ---------------------------------------------------------
# *** GESTION DES STATS ***
# ---------------------------------------------------------


print("\n🔍 Début de la comparaison des modèles...")

# 1. Charger stats existantes
if os.path.exists(net_stats_file):
    with open(net_stats_file, "r") as f:
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
with open(net_stats_file, "w") as f:
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
    model_path = f'saved_models/best_net_model_{best_model_name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n💾 Meilleur modèle sauvegardé: {model_path}")
except Exception as e:
    print(f"⚠️ Impossible de sauvegarder le modèle: {e}")



# ---------------------------
# Fonctions d’extraction/features + prédiction
# ---------------------------

def extract_features_from_net(query):
    query_upper = query.upper()
    query_len = len(query)
    word_count = len(query.split())

    # Détection d'attaques
    has_union = 'UNION' in query_upper
    has_blind = any(word in query_upper for word in ['WAITFOR', 'SLEEP', 'DELAY'])
    has_or_injection = ' OR ' in query_upper and ('1=1' in query_upper or 'EXISTS' in query_upper)
    has_subquery = '(' in query and 'SELECT' in query_upper

    # Comptage de mots-clés suspects
    suspicious_keywords = ['UNION', 'OR', 'AND', 'SELECT', 'FROM', 'WHERE', 'WAITFOR', 'SLEEP', 'DELAY', 'EXISTS']
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword in query_upper)

    # 1. BYTE_RATE_LOGNORM
    if has_union:
        byte_rate = 0.090 + min(query_len / 2000.0, 0.200)
    elif has_blind:
        byte_rate = 0.060 + min(query_len / 2500.0, 0.100)
    elif has_or_injection:
        byte_rate = 0.080 + min(keyword_count / 50.0, 0.100)
    else:
        byte_rate = 0.025 + min(query_len / 6000.0, 0.030)

    # 2. PKT_RATE_LOGNORM
    if has_blind:
        pkt_rate = 0.010 + min(word_count / 200.0, 0.005)
    elif has_union:
        pkt_rate = 0.002 + min(query_len / 8000.0, 0.003)
    elif has_or_injection or has_subquery:
        pkt_rate = 0.003 + min(keyword_count / 100.0, 0.005)
    else:
        pkt_rate = 0.001 + min(word_count / 500.0, 0.002)

    # 3. DURATION_NORM
    if has_blind:
        duration = 0.85 + min(query_len / 800.0, 0.10)
    elif has_union:
        duration = 0.150 + min(word_count / 100.0, 0.150)
    elif has_or_injection:
        duration = 0.300 + min(keyword_count / 50.0, 0.250)
    else:
        complexity = word_count + query.count('(') + query.count('JOIN')
        duration = 0.200 + min(complexity / 30.0, 0.180)

    # 4. AVG_PACKET_SIZE_NORM
    if has_union:
        avg_packet_size = 0.100 + min(query_len / 1200.0, 0.150)
    elif has_blind:
        avg_packet_size = 0.015 + min(word_count / 200.0, 0.030)
    elif has_or_injection:
        avg_packet_size = 0.050 + min(keyword_count / 40.0, 0.080)
    else:
        avg_packet_size = 0.020 + min(query_len / 2000.0, 0.030)

    print(f"Extracted features: BYTE_RATE_LOGNORM: {byte_rate}, PKT_RATE_LOGNORM: {pkt_rate}, DURATION_NORM: {duration}, AVG_PACKET_SIZE_NORM: {avg_packet_size}")  # Debugging line

    return {
        "BYTE_RATE_LOGNORM": min(byte_rate, 1.0),
        "PKT_RATE_LOGNORM": min(pkt_rate, 1.0),
        "DURATION_NORM": min(duration, 1.0),
        "AVG_PACKET_SIZE_NORM": min(avg_packet_size, 1.0)
    }

# ---------------------------

def net_predict(query_features, threshold=0.5):

    features = [query_features[col] for col in all_feature_names]

    # Conversion des features en DataFrame
    df = pd.DataFrame([features], columns=all_feature_names)

    # Obtenir la probabilité pour la classe malveillante (1)
    prob = best_model.predict_proba(df)[0][1]  # probabilité pour "malveillant"
    
    if prob < 0.05:
        prediction = 1  # Malveillant
    elif prob >= 0.05:
        prediction = 0  # Bénin
    else:
        # Utiliser un seuil intermédiaire si nécessaire
        prediction = 0 if prob >= threshold else 1
    
    return prediction

# ---------------------------

def net_predict_proba(query_features):
    if isinstance(query_features, dict):
        features = [query_features[col] for col in all_feature_names]
    elif isinstance(query_features, (list, np.ndarray)):
        if len(query_features) != len(all_feature_names):
            return 0.5  # Valeur neutre si erreur
        features = query_features
    else:
        return 0.5

    df = pd.DataFrame([features], columns=all_feature_names)
    return float(best_model.predict_proba(df)[0][1])

# ---------------------------

def net_predict_from_query(query, threshold=0.5):
    features = extract_features_from_net(query)
    return net_predict(features, threshold=threshold), net_predict_proba(features)

# ---------------------------
# Exemple de requêtes net à tester
# ---------------------------

test_queries = [
    # 1) ✅ Requêtes normales (benign)
    ("SELECT * FROM users WHERE id = 1;", 0),
    ("SELECT name FROM products WHERE category = 'electronics';", 0),
    ("SELECT COUNT(*) FROM orders WHERE status = 'shipped';", 0),

    # 2) 🚨 Requêtes dangereuses simples (facile à relier à BYTE_RATE ou DURATION)
    ("SELECT * FROM users WHERE id = 1; WAITFOR DELAY '0:0:5';", 1),                   # Blind injection → durée longue
    ("SELECT * FROM products WHERE id = 1 UNION SELECT username FROM admins;", 1),     # UNION → byte rate élevé 
    ("1 UNION SELECT password FROM users;", 1),                                        # UNION  → gros débit

    # 3) ⚠️ Requêtes subtilement anormales
    ("SELECT * FROM clients WHERE id = 1 OR EXISTS (SELECT * FROM admins);", 1),
    ("SELECT * FROM orders WHERE id = 1 AND SLEEP(2);", 1),  # SLEEP ~ DELAY

    # 4) ❓ Requêtes normales mais complexes (faux positifs possibles)
    ("SELECT name FROM products WHERE name LIKE '%phone%' AND price < 500;", 0),
    ("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id;", 0),
    ("SELECT * FROM logs WHERE action = 'login' AND time > NOW() - INTERVAL 1 DAY;", 0),
    ("SELECT COUNT(DISTINCT user_id) FROM visits WHERE page = 'home';", 0)
]


"""
# Simulation de la prédiction sur les requêtes NetFlow
print("\n===== NET TEST AVEC PROBABILITÉS =====")
for query, label in test_queries:
    pred, proba = net_predict_from_query(query, threshold=0.035)

    verdict = "✅" if pred == label else "❌"
    label_txt = "🟡 MALICIOUS" if pred == 1 else "⚪ BENIGN"
    
    print(f"{verdict} Query: {query[:50]}...")
    print(f"   Proba: {proba:.4f} → Prédit: {label_txt} (Attendu: {'🟡 MALICIOUS' if label == 1 else '⚪ BENIGN'})\n")

"""
