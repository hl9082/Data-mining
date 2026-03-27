"""
=============================================================================
Program: Dendrogram_Generator.py
Author: Huy Le (hl9082)
Purpose: Analyzes supermarket shopping cart data to visualize the inherent 
         hierarchical clustering of customer purchasing behaviors.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_dendrogram(csv_filename):
    """Generates and saves a hierarchical clustering dendrogram from a CSV dataset.

    This function loads shopping cart data, aggressively filters out any column 
    acting as a record identifier (e.g., 'Guest ID', 'RecordID', 'ID'), and computes 
    a linkage matrix using the centroid method and Euclidean distance. It then 
    plots a dendrogram truncated to show only the final 20 merged clusters and 
    saves the resulting plot as a PNG file.

    Args:
        csv_filename (str): The file path to the CSV dataset containing the 
            shopping cart data.

    Returns:
        None: The function saves a plot to the local directory and displays it, 
        but does not return any values.

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found in the directory.
        ValueError: If the dataset is empty or cannot be processed by the linkage function.
    """
    # 1. Load the dataset
    print(f"Loading {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # 2. CRITICAL TRAP: Drop the Guest ID / Record ID column!
    # The professor explicitly warned not to include this in calculations.
    id_cols = ['Guest ID', 'Record ID', 'GuestID', 'RecordID', 'ID']
    for col in id_cols:
        if col in df.columns:
            print(f"Dropping identifier column: {col}")
            X = df.drop(col, axis=1)
            break
    else:
        X = df # If no ID column is found, proceed with all data
        
    # 3. Compute the linkage matrix
    # The assignment specifies using Euclidean distance and the center of mass (centroid)
    print("Computing linkage matrix using centroid method and euclidean distance...")
    Z = linkage(X, method='centroid', metric='euclidean')
    
    # 4. Plot the dendrogram
    plt.figure(figsize=(12, 8))
    plt.title('Hierarchical Clustering Dendrogram (Top 20 Clusters)')
    plt.xlabel('Cluster Size (Number of records in the merged cluster)')
    plt.ylabel('Euclidean Distance between Center of Mass')
    
    # We use truncate_mode='lastp' and p=20 to only show the final 20 clusters
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=20,                   # show top 20 clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_leaf_counts=True,  # Shows the number of items in each cluster
        show_contracted=True
    )
    
    plt.tight_layout()
    plt.savefig('Top_20_Dendrogram.png')
    print("Success! Dendrogram saved as 'Top_20_Dendrogram.png'")
    plt.show()

if __name__ == "__main__":
    # Ensure the exact filename matches the one downloaded
    generate_dendrogram('Data/HW_CLUSTERING_SHOPPING_CART_v2255a.csv')