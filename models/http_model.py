import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cf1 = pd.read_csv("dataset_http.csv")
X = cf1["content"].astype(str)
y = cf1["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

model_accuracy = float(f"{accuracy * 100:.2f}")

def predict_query(query):
    query_vector = vectorizer.transform([query])
    return int(model.predict(query_vector)[0])
