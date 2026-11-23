import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("car_data.csv")

print("Dataset Loaded:\n")
print(df.head())

# ==============================
# PREPROCESSING
# ==============================

# Remove unwanted column if exists
if 'Unnamed: 0' in df.columns:
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Drop unnecessary columns
df.drop(['car_name', 'brand', 'model'], axis=1, inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ==============================
# PREDICTION
# ==============================
pred = model.predict(X_test)

print("\nModel Accuracy:", r2_score(y_test, pred))

# ==============================
# GRAPH
# ==============================
plt.scatter(y_test, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# ==============================
# SAVE MODEL
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
print("\nModel saved successfully!")

# ==============================
# TOP 5 RECOMMENDATION
# ==============================

df2 = pd.read_csv("car_data.csv")

if 'Unnamed: 0' in df2.columns:
    df2.drop(['Unnamed: 0'], axis=1, inplace=True)

# Score calculation
df2['Score'] = (
    (1 / df2['selling_price']) * 0.5 +
    (1 / df2['km_driven']) * 0.3 +
    (1 / df2['vehicle_age']) * 0.2
)

# Get Top 5
top5 = df2.sort_values(by='Score', ascending=False).head(5)

print("\n==============================")
print("Top 5 Cars to Buy:")
print("==============================")

print(top5[['car_name', 'selling_price', 'vehicle_age', 'km_driven']].to_string(index=False))