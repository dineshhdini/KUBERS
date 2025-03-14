import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


data = {
    "log_message": [
        "Pod crashed due to memory leak",
        "Node pressure high, eviction threshold reached",
        "Container restarted unexpectedly",
        "OOMKilled: Container ran out of memory",
        "API server timeout, request throttled",
        "Disk pressure warning on node",
        "Pod eviction due to CPU resource exhaustion",
        "Network policy blocking external access",
        "Kubelet error: Failed to pull image",
        "Persistent Volume claim failed"
    ],
    "error_type": [1, 2, 1, 3, 4, 2, 3, 5, 6, 7]  
}

df = pd.DataFrame(data)
print("Sample Kubernetes Logs Dataset:\n", df.head())


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["log_message"])
y = df["error_type"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/k8s_error_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")


def predict_k8s_error(log_message):
    model = joblib.load("models/k8s_error_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    log_vectorized = vectorizer.transform([log_message])
    prediction = model.predict(log_vectorized)
    return prediction[0]


sample_log = "Container restarted due to unexpected failure"
predicted_error = predict_k8s_error(sample_log)
print(f"Predicted Error Type: {predicted_error}")

