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

model_dir = "saved_models/all_models"
stats_file = "saved_models/model_stats.json"

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

all_feature_names = [
    "LONGUEUR", "SCORE_INJECTION", "NB_KEYWORDS", "NB_SPECIAL_CHARS", "NB_QUOTES",
    "NB_COMMENT_SYNTAX", "RATIO_SCORE_LONGUEUR", "SCORE_COMPLEXITE", "CONTIENT_OR", "CONTIENT_QUOTE",
    "CONTIENT_COMMENT", "CONTIENT_UNION", "CONTIENT_EQUAL", "CONTIENT_PARENTHESES",
    "CONTIENT_TIME", "CONTIENT_FUNCTION", "CONTIENT_IN_CLAUSE", "CLASSE_LONGUEUR",
    "LONGUEUR_NORM", "SCORE_INJECTION_NORM", "NB_KEYWORDS_NORM", "NB_SPECIAL_CHAR_NORM",
    "NB_QUOTES_NORM", "NB_COMMENTS_NORM", "RATIO_SCORE_LONGUEUR_NORM", "SCORE_COMPLEXITE_NORM",
    "CONTIENT_EXEC", "CONTIENT_SEMICOLON", "CONTIENT_UNION_SELECT"
]

MODELS = ["randomforest", "svm", "logisticregression"]
models_already_trained = (
    os.path.exists(stats_file)
    and all(os.path.exists(f"{model_dir}/{m}_model.pkl") for m in MODELS)
)

print("🔄 Chargement des datasets Vue SQL...")

# 1. Charger et concaténer tous les fichiers (on enlève les features ajoutées si besoin)
try:
    df_train = pd.read_csv("SQL_Injec_NormTrain.csv")
    df_valid = pd.read_csv("SQL_Injec_Valid.csv")
    df_test = pd.read_csv("SQL_Injec_Test.csv")

    all_data = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    all_data_reduced = all_data[["LABEL"] + selected_features]

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
feature_columns = selected_features  # on s'assure de garder le bon ordre partout

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

print("\n🔍 Début de la comparaison des modèles...")

if models_already_trained:
    print("✅ Tous les modèles sont déjà entraînés. Chargement depuis les fichiers sauvegardés...")
    results = {}
    with open(stats_file, 'r') as f:
        saved_stats = json.load(f)

    for model_name in MODELS:
        try:
            path = f"{model_dir}/{model_name}_model.pkl"
            with open(path, 'rb') as f:
                model_loaded = pickle.load(f)

            info = saved_stats.get(model_name, {})
            results[model_name.capitalize()] = {
                'model': model_loaded,
                'train_accuracy': info.get('train_accuracy', 0) / 100,
                'val_accuracy': info.get('val_accuracy', 0) / 100,
                'training_time': info.get('training_time', 0),
                'best_params': info.get('best_params', {}),
                'cv_score': info.get('cv_score', 0) / 100
            }

            print(f"✅ {model_name.capitalize()} chargé depuis {path}")
        except Exception as e:
            print(f"❌ Erreur chargement {model_name}: {e}")

    best_model_name = max(results, key=lambda m: results[m]['val_accuracy'])
    best_model = results[best_model_name]['model']
    best_score = results[best_model_name]['val_accuracy']
    train_accuracy = results[best_model_name]['train_accuracy']
    val_accuracy = results[best_model_name]['val_accuracy']
    training_time = results[best_model_name]['training_time']

else:
        for model_name, model in models.items():
            print(f"\n🔧 Test du modèle: {model_name}")

        # Validation croisée sur X_trainval, y_trainval
        grid_search = GridSearchCV(model, param_grid=params[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)

        start_time = time.time()
        grid_search.fit(X_trainval, y_trainval)
        model_training_time = time.time() - start_time

        # Score sur l'ensemble d'entraînement (juste pour info)
        train_acc = grid_search.score(X_trainval, y_trainval)
        # Moyenne des scores validation croisée
        val_acc = grid_search.best_score_

        # Évaluation du modèle sur le test final
        y_pred_test = grid_search.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Les %
        results[model_name] = {
            'model': grid_search,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': model_training_time,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }

        try:
            os.makedirs(model_dir, exist_ok=True)
            model_file = f'{model_dir}/{model_name.lower()}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(grid_search, f)
            print(f"💾 Modèle {model_name} sauvegardé dans {model_file}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde de {model_name}: {e}")

        print(f"  ⏱️ Temps: {model_training_time:.2f}s")
        print(f"  🏆 CV Score (val_acc): {val_acc:.4f}")
        print(f"  📊 Train Score: {train_acc:.4f}")
        print(f"  🔧 Meilleurs params: {grid_search.best_params_}")

        if val_acc > best_score:
            best_score = val_acc
            best_model = grid_search
            best_model_name = model_name

        # Exporter les infos sur Json
        try:
            os.makedirs('saved_models', exist_ok=True)
            stats_output = {}
            for name, info in results.items():
                stats_output[name.lower()] = {
                    'train_accuracy': round(info['train_accuracy'] * 100, 2),
                    'val_accuracy': round(info['val_accuracy'] * 100, 2),
                    'test_accuracy': round(info['test_accuracy'] * 100, 2),
                    'precision': round(info['precision'] * 100, 2),
                    'recall': round(info['recall'] * 100, 2),
                    'f1_score': round(info['f1_score'] * 100, 2),
                    'cv_score': round(info['cv_score'] * 100, 2),
                    'training_time': round(info['training_time'], 2),
                    'best_params': info['best_params']
                }

            with open(stats_file, 'w') as f:
                json.dump(stats_output, f, indent=4)
            print(f"\n📊 Statistiques de tous les modèles sauvegardées dans: {stats_file}")
        except Exception as e:
            print(f"⚠️ Erreur de sauvegarde des statistiques des modèles: {e}")

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

# Variables pour demo.py
train_accuracy = results[best_model_name]['train_accuracy']
val_accuracy = results[best_model_name]['val_accuracy']
training_time = results[best_model_name]['training_time']

exported_feature_columns = feature_columns
best_model_name = best_model_name

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

def extract_features_from_query(query):
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
    contient_in_clause = int(" in " in query)
    contient_exec = int("exec" in query or "xp_cmdshell" in query)
    contient_semicolon = int(";" in query)
    contient_union_select = int("union select" in query)

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
        contient_time, contient_function, contient_in_clause, classe_longueur,
        longueur_norm, score_injection_norm, nb_keywords_norm, nb_special_norm,
        nb_quotes_norm, nb_comments_norm, ratio_score_norm, score_complexite_norm,
        contient_exec, contient_semicolon, contient_union_select
    ]

    # On retourne toutes les features (29)
    return features

def sql_predict(query_features):
    # Prend une liste OU un dict : toujours filtrer vers selected_features
    if isinstance(query_features, dict):
        features_selected = [query_features[name] for name in selected_features]
    elif isinstance(query_features, list) or isinstance(query_features, np.ndarray):
        # S'il y a 29 features, on les mappe, sinon on suppose déjà réduit
        if len(query_features) == len(all_feature_names):
            features_dict = dict(zip(all_feature_names, query_features))
            features_selected = [features_dict[name] for name in selected_features]
        elif len(query_features) == len(selected_features):
            features_selected = query_features
        else:
            print(f"❌ Erreur: Attendu {len(selected_features)} features, reçu {len(query_features)}")
            return 0
    else:
        print("⚠️ Attention: Ce modèle nécessite des features extraites, pas une requête brute")
        return 0

    df = pd.DataFrame([features_selected], columns=selected_features)
    return int(best_model.predict(df)[0])

def sql_predict_proba(query_features):
    # Même logique que ci-dessus
    if isinstance(query_features, dict):
        features_selected = [query_features[name] for name in selected_features]
    elif isinstance(query_features, list) or isinstance(query_features, np.ndarray):
        if len(query_features) == len(all_feature_names):
            features_dict = dict(zip(all_feature_names, query_features))
            features_selected = [features_dict[name] for name in selected_features]
        elif len(query_features) == len(selected_features):
            features_selected = query_features
        else:
            return 0.5
    else:
        return 0.5
    df = pd.DataFrame([features_selected], columns=selected_features)
    return max(best_model.predict_proba(df)[0])

def sql_predict_from_query(query):
    features_all = extract_features_from_query(query)
    features_dict = dict(zip(all_feature_names, features_all))
    features_selected = [features_dict[name] for name in selected_features]
    return sql_predict(features_selected)


# Pour tester
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

# Exemples de requêtes à tester
normal_queries = [
    "SELECT name, age FROM users WHERE id = 5;",
    "UPDATE products SET price = 19.99 WHERE product_id = 120;"
]
malicious_queries = [
    "SELECT * FROM users WHERE username = 'admin' --' AND password = '123';",
    "SELECT * FROM students WHERE name = '' OR 1=1 --';"
]

def print_prediction(query):
    features_all = extract_features_from_query(query)
    features_dict = dict(zip(all_feature_names, features_all))
    print(f"Features pour {query} :")
    for k in selected_features:
        print(f"{k}: {features_dict[k]}")
    features_selected = [features_dict[name] for name in selected_features]
    pred = sql_predict(features_selected)
    label = "MALICIOUS" if pred == 1 else "BENIGN"
    print(f"Requête : {query}\n→ Prédiction du modèle : {label}\n")


print("=== Requêtes BENIGNES ===")
for q in normal_queries:
    print_prediction(q)

print("=== Requêtes MALICIEUSES ===")
for q in malicious_queries:
    print_prediction(q)
