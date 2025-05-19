#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:09:07 2025

@author: hashim.umarji
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

# Load the final scored data
final_scores_df = pd.read_csv('data/League_Normalised_RB_Scoring_ALL.csv.csv')

# Convert 'birthdate' to datetime
final_scores_df['birthdate'] = pd.to_datetime(final_scores_df['birthdate'], errors='coerce')

# Calculate age based on today's date
today = pd.to_datetime("today")
final_scores_df['Age'] = final_scores_df['birthdate'].apply(lambda x: (today - x).days // 365 if pd.notnull(x) else 0)


X = final_scores_df[[
    "Build_Up_Passing_Score", "Defensive_Ability_Score", "Final_Third_Quality_Score"
]]

# Drop rows where any of the selected score features are NaN
final_scores_df = final_scores_df.dropna(subset=X.columns)

# Re-select X now that final_scores_df is cleaned
X = final_scores_df[[
    "Build_Up_Passing_Score", "Defensive_Ability_Score", "Final_Third_Quality_Score"
]]


# Scale (standardise)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to decide optimal k
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertias, marker='o')
plt.title("Elbow Method for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Final clustering (choose k based on elbow)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
final_scores_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save to CSV
final_scores_df.to_csv("data/Final_RWB_Scores_and_Clusters.csv", index=False)

# ===========================
# PCA + Plotly Visualisation of Clusters
# ===========================

# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to DataFrame
final_scores_df["PC1"] = X_pca[:, 0]
final_scores_df["PC2"] = X_pca[:, 1]

# Plot clusters using Plotly
fig = px.scatter(
    final_scores_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=["playerName", "squadName", "Overall_RWB_Score", "Cluster"],
    title="Right Wing-Back Clusters (PCA Projection)",
    width=1000,
    height=600,
    color_continuous_scale=px.colors.qualitative.Set2
)

fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(showlegend=True)
fig.show()
fig.write_html('outputs/all_clusters.html')

cluster_0_df = final_scores_df[(final_scores_df["Cluster"] == 0) & (final_scores_df["Age"] < 28)]

fig_cluster0 = px.scatter(
    cluster_0_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=["playerName", "squadName", "Overall_RWB_Score"],
    title="Cluster 3 – Right Wing-Back Profiles (PCA Projection)",
    width=1000,
    height=600,
    color_discrete_sequence=["#FF6F61"]  # red-ish for visibility
)

fig_cluster0.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig_cluster0.update_layout(showlegend=False)
fig_cluster0.show()
fig_cluster0.write_html('outputs/cluster0.html')

# Preview top performers
print(final_scores_df.sort_values(by="Overall_RWB_Score", ascending=False).head())

