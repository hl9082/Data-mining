"""
=============================================================================
Program: PCA_KMeans.py
Author: Huy Le (hl9082)
Purpose: Projects 20D shopping cart data down to 2D PCA space, runs K-Means 
         clustering (k=4), extracts the 2D centers of mass, and plots them.
=============================================================================
"""

# WHY WE USE PANDAS: 
# Pandas is the industry standard for tabular data manipulation in Python. 
# Across this assignment, we use it to easily load the shopping cart CSV 
# file into a "DataFrame", clean the data, and efficiently filter out 
# unwanted "ID" columns before running any mathematical operations.
import pandas as pd

# WHY WE USE NUMPY:
# NumPy is Python's core library for numerical computing and linear algebra. 
# We use it extensively in this project to handle complex math that standard 
# Python cannot do natively, such as computing the covariance matrix, extracting 
# eigenvalues/eigenvectors, and performing matrix dot-product projections.
import numpy as np

# WHY WE USE MATPLOTLIB:
# The assignment requires us to generate visualizations (like the cumulative 
# sum of normalized eigenvalues and a 2D scatter plot) without using 
# proprietary software. Matplotlib is an open-source Python plotting 
# library that perfectly handles these required visual outputs.
import matplotlib.pyplot as plt

# WHY WE USE SCIKIT-LEARN (KMeans):
# Scikit-learn is 
# the most robust machine learning package in Python, and its KMeans module 
# allows us to easily compute our shopper clusters using Euclidean distances.
from sklearn.cluster import KMeans

def run_kmeans_in_pca_space(csv_filename: str, k: int = 4) -> None:
    """
    Projects data into 2D PCA space and applies K-Means clustering.

    This function centers the dataset and projects it using the top two 
    eigenvectors derived from the covariance matrix. It then applies K-Means 
    clustering to identify distinct customer groups within the 2D plane. 
    Finally, it prints the 2D (X, Y) coordinates of each cluster's center of 
    mass and generates a color-coded scatter plot.

    Args:
        csv_filename (str): The file path to the CSV dataset.
        k (int): The number of clusters to form. Defaults to 4.

    Returns:
        None: Outputs the 2D vectors directly to the console and saves a 
        scatter plot ('KMeans_PCA_2D.png') locally.
    """
    # Read the CSV file into a Pandas DataFrame.
    print(f"Loading {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # 1. Drop the ID column
    # The dataset contains ID columns (like Guest ID) which are not grocery attributes.
    # If we leave them in, the math will treat the ID number as a purchase amount!
    # We use a list comprehension to find any column with 'ID', 'GUEST', or 'RECORD' 
    # in its name, and drop them so we are left with a purely 20x20 attribute space.
    id_cols = [col for col in df.columns if 'ID' in col.upper() or 'GUEST' in col.upper() or 'RECORD' in col.upper()]
    df_clean = df.drop(columns=id_cols)
    
    # 2. Mean Center the Data (Crucial for proper PCA projection)
    data_mean = df_clean.mean()
    df_centered = df_clean - data_mean
    
    # 3. Compute Covariance and Eigenvectors
    cov_matrix = df_clean.cov()
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 4. Project onto Top 2 Eigenvectors
    print("Projecting data down to 2-Dimensional PCA Space...")
    top_2_eigenvectors = eigenvectors[:, :2]
    pca_data = np.dot(df_centered, top_2_eigenvectors)
    df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    
    # 5. Run K-Means Clustering
    print(f"\nRunning K-Means Clustering with k={k}...")
    # random_state ensures we get the exact same clusters every time we run the script
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PC1', 'PC2']])

    #printing out top 2 eigenvectors
    print("Eigenvector 1 (PC1):")
    # We use [:, 0] to get the first column, and tolist() to print it flat
    print(np.round(eigenvectors[:, 0], 3).tolist())
    
    print("\nEigenvector 2 (PC2):")
    # We use [:, 1] to get the second column
    print(np.round(eigenvectors[:, 1], 3).tolist())
    
    # 6. Extract and Print the 2D Centers of Mass
    centers_2d = kmeans.cluster_centers_
    # print("\n--- HOMEWORK ANSWER (QUESTION 7) ---")
    print("2D Centers of Mass for the 4 Clusters:")
    for i, center in enumerate(centers_2d):
        print(f"Cluster {i+1}: X (PC1) = {center[0]:.2f}, Y (PC2) = {center[1]:.2f}")
    
    # 7. Generate Scatter Plot
    print("\nGenerating Scatter Plot...")
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(k):
        cluster_points = df_pca[df_pca['Cluster'] == i]
        plt.scatter(cluster_points['PC1'], cluster_points['PC2'], 
                    c=colors[i], label=f'Cluster {i+1}', alpha=0.5, edgecolors='w', s=50)

    # Plot the black "X" marks for the centroids
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='X', s=250, label='Centroids')
    
    plt.title(f'K-Means Clustering in 2D PCA Space (k={k})', fontsize=16)
    plt.xlabel('Principal Component 1 (X)', fontsize=12)
    plt.ylabel('Principal Component 2 (Y)', fontsize=12)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('KMeans_PCA_2D.png', dpi=300)
    print("Success! Scatter plot saved to 'KMeans_PCA_2D.png'")

if __name__ == "__main__":
    run_kmeans_in_pca_space('HW_CLUSTERING_SHOPPING_CART_v2255f.csv', k=4)