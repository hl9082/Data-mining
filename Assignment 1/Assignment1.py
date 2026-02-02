"""
Author: Huy Le (hl9082)
Assignment: 1
"""
import pandas as pd
import numpy as np
import random
import os

# ==========================================
# 1. SETUP & REPRODUCIBILITY
# ==========================================
random.seed(42)
np.random.seed(42)

# ==========================================
# 2. DATA LOADING (via KaggleHub)
# ==========================================
def load_and_clean_data():
    """
    Loads the Credit Risk dataset from the local directory, renames columns to match
    assignment logic, and filters for relevant features.

    @return df - A Pandas DataFrame containing the cleaned and renamed dataset.
    """
    csv_file = 'credit_risk_dataset.csv'
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"ERROR: Could not find '{csv_file}' in the current folder. "
            "Please download the CSV from Kaggle and place it next to this script."
        )
    
    print(f"--- Loading '{csv_file}' from local directory ---")
    df = pd.read_csv(csv_file)
    
    # Rename columns to match assignment domain concepts
    rename_map = {
        'person_income': 'Income',
        'person_home_ownership': 'Home',
        'person_age': 'Age',
        'loan_int_rate': 'Interest_Rate', 
        'loan_amnt': 'Loan_Amount',
        'loan_status': 'Default'
    }
    df = df.rename(columns=rename_map)
    
    # Filter columns
    df = df[['Income', 'Home', 'Age', 'Interest_Rate', 'Loan_Amount', 'Default']]
    
    return df

# ==========================================
# 3. PREPROCESSING FUNCTIONS
# ==========================================

def discretize_feature(dataframe, column_name, num_bins=4):
    """
    Converts a continuous numerical attribute into ordinal categories using
    equal-frequency binning (quantiles). Rubric Requirement: Exactly 4 bins.

    @param dataframe - The pandas DataFrame containing the data.
    @param column_name - The name of the column to discretize.
    @param num_bins - The number of bins to create (default is 4).
    @return dataframe - The DataFrame with the specified column modified.
    """
    # qcut = Equal Frequency (Quantile Cut)
    # duplicates='drop' handles edge cases where multiple people have identical values
    dataframe[column_name] = pd.qcut(dataframe[column_name], q=num_bins, labels=False, duplicates='drop')
    return dataframe

def binarize_data(dataframe, target_col):
    """
    Transforms nominal attributes into binary attributes. Uses boolean mapping
    for 2-value attributes and One-Hot Encoding for >2 values.

    @param dataframe - The pandas DataFrame to process.
    @param target_col - The name of the target column (to exclude from processing).
    @return df_processed - A new DataFrame with nominal features binarized.
    """
    df_processed = dataframe.copy()
    feature_cols = [c for c in df_processed.columns if c != target_col]
    
    for col in feature_cols:
        # Only process object/categorical types
        if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
            unique_values = df_processed[col].unique()
            
            # Binary Attribute Rule: Map to 0/1
            if len(unique_values) == 2:
                mapper = {unique_values[0]: 0, unique_values[1]: 1}
                df_processed[col] = df_processed[col].map(mapper)
                
            # Nominal Attribute Rule: One-Hot Encode
            elif len(unique_values) > 2:
                dummies = pd.get_dummies(df_processed[col], prefix=col, dtype=int)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)
                
    return df_processed

# ==========================================
# [REMOVED] TREE CLASSES (LeafNode, DecisionNode)
# [REMOVED] HUNT'S ALGORITHM (build_tree, find_best_split, etc.)
# ==========================================

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics (Precision, Recall, F1) and Confusion Matrix counts.
    Rubric: "Precision, Recall, and F1-measure correctly calculated."

    @param y_true - List of actual class labels.
    @param y_pred - List of predicted class labels.
    @return - A dictionary containing TP, TN, FP, FN, Precision, Recall, and F1.
    """
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}

def print_confusion_matrix(metrics):
    """
    Displays the confusion matrix in a formatted grid.
    Rubric: "Confusion matrix correctly constructed... and displayed."

    @param metrics - Dictionary containing TP, TN, FP, FN counts.
    """
    print("\n--- Confusion Matrix ---")
    print(f"{'':<12} {'Pred: 0':<10} {'Pred: 1':<10}")
    print(f"{'Actual: 0':<12} {metrics['TN']:<10} {metrics['FP']:<10}")
    print(f"{'Actual: 1':<12} {metrics['FN']:<10} {metrics['TP']:<10}")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Load Data
    try:
        df = load_and_clean_data()
    except Exception as e:
        print(e)
        exit()

    target_col = 'Default'
    
    # 2. Define Attribute Types
    # Based on Kaggle Dataset:
    # Interest_Rate and Age are continuous. Home is Nominal.
    continuous_cols = ['Income', 'Age', 'Interest_Rate', 'Loan_Amount']
    
    print("\n[Step 1] Preprocessing...")
    
    # A. Discretize Continuous
    for col in continuous_cols:
        # Some columns might have NaNs (Interest_Rate). qcut handles NaNs by ignoring them
        # but we need to ensure we don't crash.
        try:
            df = discretize_feature(df, col, num_bins=4)
            print(f" -> Discretized '{col}' (4 bins)")
        except Exception as e:
            print(f" -> Warning: Could not discretize '{col}': {e}")

    # B. Binarize Nominal
    df = binarize_data(df, target_col)
    print(f" -> Binarization Complete. New Shape: {df.shape}")
    
    # 3. Split Data (90% Train, 10% Test)
    # Shuffle first
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.9 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"\n[Step 2] Splitting Complete.")
    print(f" -> Training Records: {len(train_df)}")
    print(f" -> Testing Records:  {len(test_df)}")
    
    # ---------------------------------------------------------
    # TREE CONSTRUCTION LOGIC REMOVED
    # ---------------------------------------------------------
    print("\n[Note] Tree Construction logic has been removed.")
    
    print("\n[Step 3] Evaluation (Random Baseline Only)...")
    
    # A. Actual Values
    actuals = test_df[target_col].tolist()
    
    # B. Random Weighted Baseline
    # "Compare... with a simple random model weighted based on probability"
    prob_default = train_df[target_col].mean()
    print(f" -> Training Data Default Probability: {prob_default:.2%}")
    
    random_preds = [1 if random.random() < prob_default else 0 for _ in range(len(test_df))]
    random_metrics = calculate_metrics(actuals, random_preds)
    
    # C. Display Results
    print("\n[B] Random Weighted Baseline Results")
    print_confusion_matrix(random_metrics)
    print(f"Precision: {random_metrics['Precision']:.4f}")
    print(f"Recall:    {random_metrics['Recall']:.4f}")
    print(f"F1-Score:  {random_metrics['F1']:.4f}")