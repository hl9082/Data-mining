"""
=============================================================================
Program: Cross_Correlation_Matrix.py
Author: Huy Le (hl9082)
Purpose: Computes an n x n cross-correlation matrix for shopping cart data,
         expressly excluding the Record ID to prevent skewed calculations.
=============================================================================
"""

import pandas as pd #pandas used to compute the n x n cross-correlation matrix

# to output the confusion matrix in image form
import seaborn as sns
import matplotlib.pyplot as plt

def generate_correlation_matrix(csv_filename: str) -> None:
    """
    Loads data, removes the ID column, computes the Pearson cross-correlation,
    and saves the rounded matrix to a CSV.

    Args:
        csv_filename (str): The file path to the CSV dataset containing the 
            shopping cart data.

    Returns:
        None: The function saves the generated correlation matrix as a CSV 
        and a PNG image, printing success messages to the console.
    """
    print(f"Loading {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # 1. The Trap: Safely remove the ID column
    id_cols = ['Guest ID', 'Record ID', 'GuestID', 'RecordID', 'ID']
    for col in id_cols:
        if col in df.columns:
            print(f"Dropping identifier column: {col}")
            df = df.drop(col, axis=1)
            break
            
    # 2. Compute the cross-correlation matrix using pearson (standard) method
    print("Computing Cross-Correlation matrix...")
    corr_matrix = df.corr(method='pearson')
    
    # 3. Round to 2 digits past the decimal point per instructions
    corr_matrix_rounded = corr_matrix.round(2)
    
    # Save the output for the write-up
    output_filename = 'Cross_Correlation_Matrix.csv'
    corr_matrix_rounded.to_csv(output_filename)
    print(f"Success! Matrix saved to '{output_filename}'")

    # 4. Generate and save the Heatmap Image
    print("Generating heatmap image...")
    # Set the size of the image so the 20x20 matrix isn't cramped
    plt.figure(figsize=(14, 12))
    
    # Create the heatmap
    # cmap='coolwarm' makes negative correlations blue and positive ones red
    # annot=True writes the actual numbers inside the squares
    sns.heatmap(corr_matrix_rounded, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1.0, vmax=1.0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .8})
    
    plt.title('Shopping Cart Cross-Correlation Matrix', fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save the image to the current directory
    output_image = 'Cross_Correlation_Matrix.png'
    plt.savefig(output_image, dpi=300) # dpi=300 ensures it is high resolution for your PDF
    print(f"Success! Heatmap image saved to '{output_image}'")

if __name__ == "__main__":
    generate_correlation_matrix('Data/HW_CLUSTERING_SHOPPING_CART_v2255a.csv')