import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==============================
# STEP 1: Load Dataset
# ==============================

df = pd.read_csv("/Users/sahassrak/Downloads/Data.csv")

print("Dataset Loaded")
print(df.head())

# ==============================
# STEP 2: Convert Columns to Numeric
# ==============================

numeric_cols = ["Su", "Sy", "A5", "Bhn", "Ro"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==============================
# STEP 3: Feature Selection
# ==============================

features = ["Su", "Sy", "A5", "Bhn", "Ro"]

df = df.dropna(subset=features)
df = df.reset_index(drop=True)

X = df[features]

# ==============================
# STEP 4: Feature Scaling
# ==============================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# STEP 5: K-Means Clustering
# ==============================

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# ==============================
# STEP 6: KNN Recommendation Model
# ==============================

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X_scaled)

# ==============================
# STEP 7: User Input
# ==============================

print("\nEnter Required Mechanical Properties")

Su_input = float(input("Ultimate Tensile Strength (Su): "))
Sy_input = float(input("Yield Strength (Sy): "))
A5_input = float(input("Ductility (A5): "))
Bhn_input = float(input("Hardness (Bhn): "))
Ro_input = float(input("Density (Ro): "))

user_input = pd.DataFrame([{
    "Su": Su_input,
    "Sy": Sy_input,
    "A5": A5_input,
    "Bhn": Bhn_input,
    "Ro": Ro_input
}])

user_scaled = scaler.transform(user_input)

# ==============================
# STEP 8: Find Nearest Materials
# ==============================

distances, indices = knn.kneighbors(user_scaled)

recommended = df.iloc[indices[0]]

print("\nTop 5 Recommended Materials:\n")

for i, row in recommended.iterrows():

    print("Material:", row["Material"])
    print("Heat Treatment:", row["Heat treatment"])
    print("Cluster:", row["Cluster"])

    print("Property Differences from Requirement:")

    print("Su difference :", float(row["Su"]) - Su_input)
    print("Sy difference :", float(row["Sy"]) - Sy_input)
    print("A5 difference :", float(row["A5"]) - A5_input)
    print("Bhn difference:", float(row["Bhn"]) - Bhn_input)
    print("Ro difference :", float(row["Ro"]) - Ro_input)

    print("-" * 40)

# ==============================
# STEP 9: Cluster Visualization
# ==============================

plt.figure()

plt.scatter(df["Su"], df["Bhn"], c=df["Cluster"])

plt.xlabel("Tensile Strength (Su)")
plt.ylabel("Hardness (Bhn)")
plt.title("Material Clusters")

plt.show()