import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/adult.csv")

# Drop unused columns to avoid feature mismatch
df = df.drop(columns=['fnlwgt', 'educational-num', 'capital-gain', 'capital-loss'])

# Clean data
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Encode categorical features (including 'income')
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and label
X = df.drop("income", axis=1)
y = df["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and metadata
joblib.dump(model, "salary_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
