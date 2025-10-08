# train.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")  # change if file name differs

# 1. Load dataset
df = pd.read_csv(CSV_PATH)
print("Loaded CSV shape:", df.shape)
print("Columns:", df.columns.tolist())

# Expect columns like: N, P, K, temperature, humidity, ph, rainfall, label (or crop)
# Try to auto-detect target column
possible_targets = ["label", "crop", "Crop", "Cropname"]
target_col = None
for c in possible_targets:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # fallback: assume last column is label
    target_col = df.columns[-1]
    print("Warning: target column auto-selected as", target_col)

# Features columns (common names)
feat_candidates = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
features = [c for c in feat_candidates if c in df.columns]
print("Detected features:", features)

# If any missing, try lower/upper case alternatives
if len(features) != 7:
    # attempt some common names
    alt_map = {
        "N": ["N","Nitrogen","nitrogen"],
        "P": ["P","Phosphorus","phosphorus","Phosporus"],
        "K": ["K","Potassium","potassium"],
        "temperature": ["temperature","Temperature","temp","Temp"],
        "humidity": ["humidity","Humidity"],
        "ph": ["ph","pH","PH"],
        "rainfall": ["rainfall","Rainfall","rain"]
    }
    features = []
    for key, names in alt_map.items():
        for n in names:
            if n in df.columns:
                features.append(n)
                break
print("Final feature columns used:", features)
if len(features) != 7:
    raise RuntimeError("Could not find all 7 feature columns. Please check CSV column names.")

X = df[features].values
y_raw = df[target_col].values

# 2. Encode labels (crop names -> integers)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label classes (encoder):", list(le.classes_))

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scalers: MinMax then Standard (match your app pipeline)
mx = MinMaxScaler()
X_train_mx = mx.fit_transform(X_train)
X_test_mx = mx.transform(X_test)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_mx)
X_test_sc = sc.transform(X_test_mx)

# 5. Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_sc, y_train)

print("Train acc:", clf.score(X_train_sc, y_train))
print("Test acc:", clf.score(X_test_sc, y_test))

# 6. Save model, scalers, and label encoder
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(BASE_DIR, "minmaxscaler.pkl"), "wb") as f:
    pickle.dump(mx, f)
with open(os.path.join(BASE_DIR, "standscaler.pkl"), "wb") as f:
    pickle.dump(sc, f)
with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("Saved model.pkl, minmaxscaler.pkl, standscaler.pkl, label_encoder.pkl")
print("You can now run your Flask app and it should produce real predictions.")
