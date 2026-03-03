import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import joblib

# Load dataset (TAB separated)
data = pd.read_csv(
    r"C:\Users\LENOVO\OneDrive\Desktop\DiseasePredictionApp\disease.csv",
    sep="\t"
)

print("Dataset Loaded Successfully")
print("Columns:", data.columns)

# ----------------------------
# Separate target and features
# ----------------------------
y = data["Disease"]
X = data.drop("Disease", axis=1)

# ----------------------------
# Convert symptom columns into list format
# ----------------------------
X = X.values.tolist()

# Remove NaN values
X = [[symptom for symptom in row if pd.notna(symptom)] for row in X]

# ----------------------------
# Convert symptoms to 0/1 format
# ----------------------------
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

print("Total Unique Symptoms:", len(mlb.classes_))

# ----------------------------
# Encode disease labels
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")

# ----------------------------
# Save everything
# ----------------------------
joblib.dump((model, mlb, le), "model.pkl")

print("Model saved as model.pkl")