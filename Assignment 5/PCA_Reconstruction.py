"""
=============================================================================
Program: PCA_Reconstruction.py
Author: Huy Le (hl9082)
Purpose: Re-projects the 2D K-Means centers back into the original 20D space 
         to identify the grocery prototypes.
=============================================================================
"""

# Pandas is the industry standard for tabular data manipulation in Python. 
# Across this assignment, we use it to easily load the shopping cart CSV 
# file into a "DataFrame", clean the data, and efficiently filter out 
# unwanted "ID" columns before running any mathematical operations.
import pandas as pd
# NumPy is Python's core library for numerical computing and linear algebra. 
# We use it extensively in this project to handle complex math that standard 
# Python cannot do natively, such as computing the covariance matrix, extracting 
# eigenvalues/eigenvectors, and performing matrix dot-product projections.
import numpy as np
# Scikit-learn is the most robust machine learning package in Python, and its KMeans module 
# allows us to easily compute our shopper clusters using Euclidean distances.
from sklearn.cluster import KMeans

def run_pca_reconstruction(csv_filename: str) -> None:
    """
    Projects data to 2D PCA space, clusters it, and reconstructs the centroids.
    
    This script performs the Phase 2 operations to get the 2D cluster centers, 
    and then executes the Phase 3 'Re-Projection' by multiplying the centers 
    against the transposed top 2 eigenvectors and adding back the data mean.
    It prints the resulting 20-dimensional grocery shopping carts.
    """
    df = pd.read_csv(csv_filename)
    id_cols = [col for col in df.columns if 'ID' in col.upper() or 'GUEST' in col.upper() or 'RECORD' in col.upper()]
    df_clean = df.drop(columns=id_cols)
    
    # PCA Prep
    data_mean = df_clean.mean()
    df_centered = df_clean - data_mean
    cov_matrix = df_clean.cov()
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort and take top 2
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    top_2_eigenvectors = eigenvectors[:, :2]
    
    # 2D Projection & Clustering
    pca_data = np.dot(df_centered, top_2_eigenvectors)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)
    centers_2d = kmeans.cluster_centers_
    
    # -------------------------------------------------------------------------
    # STEP 8: RE-PROJECTION (Multiplying centers back by eigenvectors)
    # Formula: Original_Space = (PCA_Space DOT Eigenvectors.T) + Data_Mean
    # -------------------------------------------------------------------------
    reconstructed_centers = np.dot(centers_2d, top_2_eigenvectors.T) + data_mean.values
    
    reconstructed_df = pd.DataFrame(reconstructed_centers, columns=df_clean.columns)
    
    
    print("Reconstructed Prototypes (20D Centers of Mass):\n")
    print(reconstructed_df.round(1).T)
    
    # Check for the mathematical anomaly (Negative values)
    # We will reconstruct ALL individual shoppers, not just the centers, to check for negatives
    reconstructed_all_points = np.dot(pca_data, top_2_eigenvectors.T) + data_mean.values
    num_negatives = (reconstructed_all_points < 0).sum()
    print(f"\nMathematical Anomaly Check:")
    print(f"When reconstructing all individual shoppers, {num_negatives} grocery values went negative!")

if __name__ == "__main__":
    run_pca_reconstruction('HW_CLUSTERING_SHOPPING_CART_v2255f.csv')