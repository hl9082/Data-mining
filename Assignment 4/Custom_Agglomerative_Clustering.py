"""
=============================================================================
Program: Custom_Agglomerative_Clustering.py
Author: Huy Le (hl9082)
Purpose: Implements Agglomerative Clustering entirely from scratch using the 
         Centroid (Center of Mass) method and Euclidean Distance.
=============================================================================
"""

import pandas as pd
import numpy as np
import time

def calculate_euclidean_distance(centroid_a, centroid_b):
    """
    Calculates the Euclidean distance between two cluster centroids.
    
    Args:
        centroid_a (numpy.ndarray): An array representing the center of mass 
            (average attributes) of the first cluster.
        centroid_b (numpy.ndarray): An array representing the center of mass 
            (average attributes) of the second cluster.

    Returns:
        float: The exact Euclidean distance between the two centroids.

    """
    # numpy.linalg.norm computes the exact Euclidean distance formula
    return np.linalg.norm(centroid_a - centroid_b)

def run_custom_agglomeration(csv_filename):
    """
    This function iteratively groups records by finding the two active clusters 
    with the shortest Euclidean distance. This algorithm intentionally avoids 
    distance caching and recomputes all inter-cluster distances on every pass. When two clusters 
    merge, the newly formed cluster inherits the lowest integer ID of its 
    parents, ensuring the final remaining cluster is labeled with the 
    lowest possible starting ID.

    Args:
        csv_filename (str): The file path to the CSV dataset containing the 
            shopping cart data.

    Returns:
        None: The function prints progress updates and the sizes of the smallest clusters during the last 20 merges 
        directly to the console.

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found in the 
            current directory.

    """
    print(f"Loading {csv_filename}...")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{csv_filename}'.")
        raise
    
    # Drop identifier columns (The Trap!)
    id_cols = ['Guest ID', 'Record ID', 'GuestID', 'RecordID', 'ID']
    for col in id_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
            break
            
    # Convert to a numpy array for faster mathematical operations
    dataset = df.to_numpy()
    total_records = len(dataset)
    
    # -------------------------------------------------------------------------
    # STEP 1: Initialization
    # We use a dictionary to track our active clusters.
    # Key: Cluster ID (integer)
    # Value: A dictionary containing the 'centroid' and 'size'
    # -------------------------------------------------------------------------
    active_clusters = {}
    for index in range(total_records):
        active_clusters[index] = {
            'centroid': dataset[index], # Starts as just the individual shopper's data
            'size': 1,                  # Starts with 1 shopper
            'members': [index]          # Tracks which original rows are in this cluster
        }
    
    
    merge_history_smallest_sizes = []
    
    print(f"Starting Optimized Agglomeration Loop with {len(active_clusters)} clusters...")
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # STEP 2: The Main Agglomeration Loop
    # -------------------------------------------------------------------------
    while len(active_clusters) > 1:
        minimum_distance = float('inf')
        cluster_pair_to_merge = (None, None)
        
        cluster_ids = list(active_clusters.keys())
        
        # Avoid comparing a cluster to itself by starting j at i + 1
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id_a = cluster_ids[i]
                id_b = cluster_ids[j]
                
                dist = calculate_euclidean_distance(
                    active_clusters[id_a]['centroid'], 
                    active_clusters[id_b]['centroid']
                )
                
                if dist < minimum_distance:
                    minimum_distance = dist
                    cluster_pair_to_merge = (id_a, id_b)
        
        # -------------------------------------------------------------------------
        # STEP 3: Merge and ID Preservation
        # -------------------------------------------------------------------------
        id_a, id_b = cluster_pair_to_merge
        cluster_a = active_clusters[id_a]
        cluster_b = active_clusters[id_b]
        
        # Record the size of the smaller cluster being merged
        smaller_size = min(cluster_a['size'], cluster_b['size'])
        merge_history_smallest_sizes.append(smaller_size)
        
        # Calculate the new Center of Mass (Weighted Average)
        total_size = cluster_a['size'] + cluster_b['size']
        new_centroid = ((cluster_a['centroid'] * cluster_a['size']) + 
                        (cluster_b['centroid'] * cluster_b['size'])) / total_size
        
        combined_members = cluster_a['members'] + cluster_b['members']
        
        # The lowest cluster label must persist!
        persisting_id = min(id_a, id_b)
        deleted_id = max(id_a, id_b)
        
        # Update the persisting cluster with the new merged data
        active_clusters[persisting_id] = {
            'centroid': new_centroid,
            'size': total_size,
            'members': combined_members
        }
        
        # Delete the higher ID cluster since it was absorbed into the smaller ID
        del active_clusters[deleted_id]
        # Print progress update every 100 merges so the user knows it hasn't frozen
        if len(active_clusters) % 100 == 0:
            print(f"Clusters remaining: {len(active_clusters)}...")

    end_time = time.time()
    print(f"\nClustering complete in {round(end_time - start_time, 2)} seconds!")
    
    # -------------------------------------------------------------------------
    # STEP 4: Reporting the results for the Write-Up
    # -------------------------------------------------------------------------
    last_20_sizes = merge_history_smallest_sizes[-20:]

    print(f"Final surviving cluster ID: {list(active_clusters.keys())[0]}")
     
    print(f"Sizes of the smallest clusters in the last 20 merges:\n{last_20_sizes}")

if __name__ == "__main__":
    run_custom_agglomeration('Data/HW_CLUSTERING_SHOPPING_CART_v2255a.csv')