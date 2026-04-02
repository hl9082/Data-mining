"""
=============================================================================
Program: Cross_Correlation_Matrix.py
Author: Huy Le (hl9082)
Purpose: Computes an n x n cross-correlation matrix for shopping cart data,
         expressly excluding the Record ID to prevent skewed calculations.
=============================================================================
"""

import pandas as pd #pandas used to compute the n x n cross-correlation matrix

def generate_correlation_matrix(csv_filename: str) -> None:
    """
    Loads data, removes the ID column, computes the Pearson cross-correlation,
    and saves the rounded matrix to a CSV.

    Args:
        csv_filename (str): The file path to the CSV dataset containing the 
            shopping cart data.

    Returns:
        None: The function saves the generated correlation matrix directly to a 
        local CSV file and prints a success message to the console.
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
            
    # 2. Compute the cross-correlation matrix
    print("Computing Cross-Correlation matrix...")
    corr_matrix = df.corr(method='pearson')
    
    # 3. Round to 2 digits past the decimal point per instructions
    corr_matrix_rounded = corr_matrix.round(2)
    
    # Save the output for the write-up
    output_filename = 'Cross_Correlation_Matrix.csv'
    corr_matrix_rounded.to_csv(output_filename)
    print(f"Success! Matrix saved to '{output_filename}'")

if __name__ == "__main__":
    generate_correlation_matrix('Data/HW_CLUSTERING_SHOPPING_CART_v2255a.csv')