import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("Crop_recommendation_FINAL_STRONG_REINF.csv")

feature_cols = ["N", "P", "K", "temperature",
                "humidity", "ph", "rainfall", "ec"]
X = data[feature_cols]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)

stand_scaler = StandardScaler()
X_train_final = stand_scaler.fit_transform(X_train_mm)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_final, y_train)

X_test_mm = minmax_scaler.transform(X_test)
X_test_final = stand_scaler.transform(X_test_mm)
y_pred = model.predict(X_test_final)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

with open("minmaxscaler.pkl", "wb") as f:
    pickle.dump(minmax_scaler, f)

with open("standscaler.pkl", "wb") as f:
    pickle.dump(stand_scaler, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New model trained successfully.")
