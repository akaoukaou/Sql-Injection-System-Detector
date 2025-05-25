import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset_sql.csv")
X = df["Query"].astype(str)
y = df["Label"]

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorisation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model = LinearSVC()
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

def predict_query(query):
    query_vec = vectorizer.transform([query])
    return int(model.predict(query_vec)[0])

model_accuracy = round(accuracy * 100, 2)
print(f"SVM accuracy: {model_accuracy}%")
