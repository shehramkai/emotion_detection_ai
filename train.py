import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train["text"]
y_train = train["label"]

X_test = test["text"]
y_test = test["label"]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(model, "emotion_model.pkl")

print("\nModel saved as emotion_model.pkl")