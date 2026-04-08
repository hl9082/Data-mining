"""
=============================================================================
Program: PCA_Analysis.py
Author: Huy Le (hl9082)
Purpose: Performs Principal Component Analysis (PCA) on shopping cart data.
         Computes the covariance matrix, eigenvalues, eigenvectors, and 
         generates a Scree Plot and Cumulative VAF graph.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_pca_phase_1(csv_filename: str) -> None:
    print(f"Loading {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # -------------------------------------------------------------------------
    # STEP 1: The Trap - Drop the ID column
    # -------------------------------------------------------------------------
    id_cols = [col for col in df.columns if 'ID' in col.upper() or 'GUEST' in col.upper() or 'RECORD' in col.upper()]
    for col in id_cols:
        print(f"Dropping identifier column to prevent skewed covariance: {col}")
        df = df.drop(col, axis=1)
        
    # -------------------------------------------------------------------------
    # STEP 2: Compute the Covariance Matrix
    # -------------------------------------------------------------------------
    print("\nComputing 20x20 Covariance Matrix...")
    cov_matrix = df.cov()
    
    # -------------------------------------------------------------------------
    # STEP 3: Compute Eigenvalues and Eigenvectors
    # -------------------------------------------------------------------------
    # We use np.linalg.eigh because a covariance matrix is symmetric, 
    # which mathematically guarantees real (non-complex) eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # By default, eigh sorts ascending. We must reverse them so the 
    # largest eigenvalues (Principal Components) are first.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nTop 5 Eigenvalues:")
    print(np.round(eigenvalues[:5], 2))
    
    # -------------------------------------------------------------------------
    # STEP 4: Calculate Variance Accounted For (VAF)
    # -------------------------------------------------------------------------
    total_variance = np.sum(eigenvalues)
    vaf = eigenvalues / total_variance
    cumulative_vaf = np.cumsum(vaf)
    
    # Find how many components are needed to reach 90% VAF
    components_90 = np.argmax(cumulative_vaf >= 0.90) + 1
    print(f"\nNumber of eigenvectors required to hit 90% VAF: {components_90}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Generate the Scree Plot and VAF Plot
    # -------------------------------------------------------------------------
    print("\nGenerating Plots...")
    plt.figure(figsize=(14, 5))
    
    # 1st Subplot: Scree Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
    plt.title('Scree Plot of Eigenvalues', fontsize=14)
    plt.xlabel('Principal Component (Eigenvector Number)', fontsize=12)
    plt.ylabel('Eigenvalue (Amount of Variance)', fontsize=12)
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2nd Subplot: Cumulative VAF Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_vaf) + 1), cumulative_vaf, marker='s', linestyle='-', color='r')
    plt.axhline(y=0.90, color='k', linestyle='--', label='90% VAF Threshold')
    plt.axvline(x=components_90, color='green', linestyle=':', label=f'Hits 90% at PC {components_90}')
    
    plt.title('Cumulative Variance Accounted For (VAF)', fontsize=14)
    plt.xlabel('Number of Principal Components Used', fontsize=12)
    plt.ylabel('Cumulative Proportion of Variance', fontsize=12)
    plt.xticks(range(1, len(cumulative_vaf) + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_filename = 'PCA_Scree_and_VAF.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Success! Graphs saved as '{plot_filename}'")

if __name__ == "__main__":
    run_pca_phase_1('HW_CLUSTERING_SHOPPING_CART_v2255f.csv')