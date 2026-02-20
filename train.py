import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import datetime

print("Decision Tree Retraining Started...")

# Load dataset
df = pd.read_csv("fraud.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=8,
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

# Log retraining
with open("retrain_log.txt", "w") as f:
    f.write(f"Last Retrained: {datetime.datetime.now()}\nAccuracy: {accuracy}")

print("Decision Tree Retraining Completed!")
