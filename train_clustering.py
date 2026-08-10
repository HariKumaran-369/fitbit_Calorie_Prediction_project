import os
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from preprocessing import (
    get_clean_data,
    encode_data,
    scale_data,
    NUMERIC_COLS,
    TARGET_COL,
)
matplotlib.use("Agg")
# KMeans (main method), and also compare with Hierarchical
# Clustering and DBSCAN.
MODEL_DIR = "models"
FIG_DIR = "reports/figures"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load data and keep Workout_Type aside
    df = get_clean_data()
    workout_type = df["Workout_Type"]

    features_df = df.drop(columns=["Workout_Type", TARGET_COL])
    features_df = encode_data(features_df)

    # Scale the numeric columns
    num_cols = [c for c in NUMERIC_COLS if c in features_df.columns]
    features_df, scaler = scale_data(features_df, num_cols)

    # Reduce to 2 dimensions using PCA so we can plot it
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_df.values)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Elbow method - try different values of K and check silhouette score
    print("Checking different numbers of clusters (K):")
    silhouette_scores = {}
    inertia_values = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_pca)
        score = silhouette_score(features_pca, labels)
        silhouette_scores[k] = score
        inertia_values[k] = km.inertia_
        print("K =", k, " Silhouette Score =", round(score, 4))

    best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
    print("Best K based on silhouette score:", best_k)

    # Plot the elbow chart and the silhouette chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        list(inertia_values.keys()),
        list(inertia_values.values()),
        marker="o",
        color="crimson",
    )
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(
        list(silhouette_scores.keys()),
        list(silhouette_scores.values()),
        marker="o",
        color="green",
    )
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs K")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "/elbow_silhouette.png")
    plt.close()

    # Fit the final KMeans model with the best K
    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(features_pca)
    kmeans_score = silhouette_score(features_pca, kmeans_labels)
    print(
        "KMeans Silhouette Score:",
        round(kmeans_score, 4),
        "(target >= 0.15)",
    )

    # Compare with Hierarchical Clustering
    hierarchical_model = AgglomerativeClustering(n_clusters=best_k)
    hierarchical_labels = hierarchical_model.fit_predict(features_pca)
    hierarchical_score = silhouette_score(features_pca, hierarchical_labels)
    print(
        "Hierarchical Clustering Silhouette Score:",
        round(hierarchical_score, 4),
    )

    # Compare with DBSCAN
    dbscan_model = DBSCAN(eps=1.2, min_samples=10)
    dbscan_labels = dbscan_model.fit_predict(features_pca)
    num_dbscan_clusters = len(set(dbscan_labels)) - (
        1 if -1 in dbscan_labels else 0
    )
    if num_dbscan_clusters >= 2:
        mask = dbscan_labels != -1
        dbscan_score = silhouette_score(
            features_pca[mask], dbscan_labels[mask]
        )
    else:
        dbscan_score = None
    print(
        "DBSCAN found",
        num_dbscan_clusters,
        "clusters. Silhouette Score:",
        dbscan_score,
    )

    # Plot all three clustering results side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, labels, title in zip(
        axes,
        [kmeans_labels, hierarchical_labels, dbscan_labels],
        ["KMeans", "Hierarchical", "DBSCAN"],
    ):
        ax.scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.6,
        )
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "/clustering_comparison.png")
    plt.close()

    # Look at what each KMeans cluster represents
    result_df = features_df.copy()
    result_df["Cluster"] = kmeans_labels
    result_df["Workout_Type"] = workout_type.values

    print("Cluster sizes:")
    print(result_df["Cluster"].value_counts())

    print("Cluster vs Workout_Type:")
    print(pd.crosstab(result_df["Cluster"], result_df["Workout_Type"]))

    result_df.to_csv("reports/cluster_interpretation.csv", index=False)

    # Save the KMeans model (our main/chosen method)
    joblib.dump(kmeans_model, MODEL_DIR + "/kmeans_model.pkl")
    joblib.dump(pca, MODEL_DIR + "/pca_model.pkl")
    joblib.dump(scaler, MODEL_DIR + "/clustering_scaler.pkl")

    print("Clustering models saved in models/ folder")


if __name__ == "__main__":
    main()
