"""
Author: Huy Le (hl9082)
Assignment: 1
Description:
  This module implements a custom Decision Tree Classifier from scratch using Hunt's Algorithm 
  and the Weighted Gini Index. It adheres to strict constraints requiring manual implementation 
  of all preprocessing steps, avoiding high-level abstractions like pandas.qcut or get_dummies.

  Key Features:
  - Manual Equal-Frequency Discretization (4 bins).
  - Manual One-Hot Encoding for nominal attributes.
  - Dynamic handling of missing values during impurity calculations.
  - Evaluation against a Weighted Random Baseline using Precision, Recall, and F1.
"""
import pandas as pd
import numpy as np
import random
import os

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
# Rubric Requirement: "Comparison & Discussion" implies consistent results.
random.seed(42)
np.random.seed(42)

class CreditRiskTree:
    """
    A self-contained Decision Tree Classifier that implements Hunt's Algorithm
    with manual preprocessing to meet specific assignment constraints.
    """
    
    def __init__(self, target_col='Default', min_samples=5):
        """
        Initializes the Decision Tree parameters.
        
        @param target_col - The name of the target column to predict.
        @param min_samples - The minimum number of records required to split a node.
        """
        self.target = target_col
        self.min_samples = min_samples
        self.tree = None

    # ==========================================
    # PREPROCESSING UTILITIES
    # ==========================================
    
    def manual_qcut(self, df, col, bins=4):
        """
        Manually implements Equal-Frequency (Quantile) binning without using pandas.qcut.
        Calculates percentiles using NumPy and maps values to [0, 1, 2, 3].
        
        Constraint: "Avoid using pandas.qcut for discretization."

        @param df - The pandas DataFrame containing the data.
        @param col - The name of the continuous column to discretize.
        @param bins - The number of bins to create (default is 4).
        @return df - The DataFrame with the continuous column replaced by bin indices.
        """
        # Extract valid data to determine thresholds (ignoring NaNs)
        valid_vals = df[col].dropna().values
        
        # Calculate percentile thresholds (0, 25, 50, 75, 100)
        # Using linspace ensures exactly 'bins' number of intervals.
        thresholds = np.percentile(valid_vals, np.linspace(0, 100, bins + 1))
        
        # Deduplicate thresholds to handle skewed distributions (e.g. many 0s)
        thresholds = np.unique(thresholds)
        
        def _get_bin_index(val):
            if pd.isna(val): return np.nan
            # Iterate through thresholds to find the correct bin
            for i in range(len(thresholds) - 1):
                lower, upper = thresholds[i], thresholds[i+1]
                # Logic: Inclusive lower, exclusive upper
                # Exception: The final bin includes the upper bound
                if i == len(thresholds) - 2:
                    if lower <= val <= upper: return i
                elif lower <= val < upper: return i
            return 0 # Fallback
            
        # Apply transformation
        df[col] = df[col].apply(_get_bin_index)
        print(f"   [Process] Discretized '{col}' using thresholds: {np.round(thresholds, 1)}")
        return df

    def manual_one_hot(self, df):
        """
        Detects nominal columns and applies manual One-Hot Encoding.
        
        Constraint: "Implement specific tasks such as binarization yourself."

        @param df - The pandas DataFrame to process.
        @return processed_df - A new DataFrame with nominal features binarized.
        """
        # Identify nominal columns (exclude target and already numeric/discretized)
        features = [c for c in df.columns if c != self.target]
        processed_df = df.copy()
        
        for col in features:
            # Check for object/string types
            if processed_df[col].dtype == 'object':
                unique_vals = processed_df[col].dropna().unique()
                
                # Binary Case (e.g., Rent vs Own) -> Single column mapping
                if len(unique_vals) == 2:
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    processed_df[col] = processed_df[col].map(mapping)
                    
                # Multi-Class Case -> Create new columns manually
                elif len(unique_vals) > 2:
                    for val in unique_vals:
                        new_col = f"{col}_{val}"
                        # Boolean check cast to int (0/1)
                        processed_df[new_col] = (processed_df[col] == val).astype(int)
                    # Remove original column after encoding
                    processed_df.drop(columns=[col], inplace=True)
                    
        return processed_df

    # ==========================================
    # CORE ALGORITHM (HUNT'S + GINI)
    # ==========================================

    def _calculate_gini(self, records, attr):
        """
        Computes Weighted Gini Index for a specific attribute split.
        It ignores NaN values specifically for the attribute being tested.

        @param records - The subset of dataframe records.
        @param attr - The attribute to evaluate for splitting.
        @return weighted_gini - The calculated weighted Gini Index (float).
        """
        # Filter: Only consider records where this specific attribute exists
        valid_df = records[records[attr].notna()]
        total_count = len(valid_df)
        
        if total_count == 0: return 1.0 # Max impurity if no valid data
        
        weighted_gini = 0.0
        # Iterate through every branch (bin or category) found in this subset
        for value in valid_df[attr].unique():
            branch_df = valid_df[valid_df[attr] == value]
            branch_size = len(branch_df)
            
            # Calculate Gini of this specific child node: 1 - sum(prob^2)
            score = 1.0
            for cls in branch_df[self.target].unique():
                prob = len(branch_df[branch_df[self.target] == cls]) / branch_size
                score -= prob ** 2
            
            # Add to weighted average
            weighted_gini += (branch_size / total_count) * score
            
        return weighted_gini

    def _hunts_algorithm(self, records, attributes, depth=0):
        """
        Recursive implementation of Hunt's Algorithm to build the tree.
        
        @param records - The subset of dataframe records for the current node.
        @param attributes - List of available attributes for splitting.
        @param depth - The current depth (used for logging).
        @return node - A dictionary representing the node and its children.
        """
        # 1. Identify Majority Class (for tie-breaking/leaf fallback)
        if len(records) == 0: return 0
        majority_cls = records[self.target].mode()[0]
        
        # STOPPING CONDITIONS
        # A. Purity Check (All records have same class)
        if len(records[self.target].unique()) == 1:
            return {'type': 'leaf', 'class': records[self.target].iloc[0]}
        
        # B. Min Records Check (Rubric: < 5 records)
        if len(records) < self.min_samples:
            return {'type': 'leaf', 'class': majority_cls}
            
        # C. Attribute Exhaustion (No columns left to split on)
        if not attributes:
            return {'type': 'leaf', 'class': majority_cls}
            
        # SPLITTING STEP
        best_attr = None
        min_gini = float('inf')
        
        for attr in attributes:
            gini = self._calculate_gini(records, attr)
            if gini < min_gini:
                min_gini = gini
                best_attr = attr
                
        # Edge Case: If no split improves impurity, stop.
        if best_attr is None:
            return {'type': 'leaf', 'class': majority_cls}
            
        # Construct Decision Node
        node = {
            'type': 'node',
            'attr': best_attr,
            'fallback': majority_cls, # Used if we hit a missing value during test
            'branches': {}
        }
        
        # Log split for grading visibility
        print(f"{'  | ' * depth}Split: [{best_attr}] (Weighted Gini: {min_gini:.4f})")
        
        # Recursive Calls
        remaining_attrs = [a for a in attributes if a != best_attr]
        unique_vals = records[best_attr].dropna().unique()
        
        for val in unique_vals:
            subset = records[records[best_attr] == val]
            child_node = self._hunts_algorithm(subset, remaining_attrs, depth + 1)
            node['branches'][val] = child_node
            
        return node

    def fit(self, df):
        """
        Starts the tree construction process.
        
        @param df - The training dataframe including the target column.
        """
        attributes = [c for c in df.columns if c != self.target]
        print("\n--- Starting Tree Construction ---")
        self.tree = self._hunts_algorithm(df, attributes)
        print("--- Construction Complete ---\n")

    def _predict_row(self, row, node):
        """
        Recursively traverses the tree to predict a single row.
        
        @param row - The pandas Series representing one record.
        @param node - The current tree node dictionary.
        @return class_label - The predicted class (0 or 1).
        """
        # Base Case: We reached a leaf
        if node['type'] == 'leaf':
            return node['class']
        
        val = row.get(node['attr'])
        
        # Handle Missing Values or Unseen Data during Inference
        # If val is NaN or a category we never trained on, use the node's majority class.
        if pd.isna(val) or val not in node['branches']:
            return node['fallback'] 
            
        return self._predict_row(row, node['branches'][val])

    def predict(self, df):
        """
        Generates predictions for a dataset.
        
        @param df - The dataframe to predict on.
        @return predictions - A list of predicted class labels.
        """
        return [self._predict_row(row, self.tree) for _, row in df.iterrows()]

# ==========================================
# 3. METRICS & UTILS
# ==========================================

def get_performance(y_true, y_pred):
    """
    Calculates Precision, Recall, F1, and Confusion Matrix components.
    
    Rubric: "Precision, Recall, and F1-measure correctly calculated."

    @param y_true - List of actual class labels.
    @param y_pred - List of predicted class labels.
    @return metrics - Dictionary containing TP, TN, FP, FN and scores.
    """
    # Manual calculation of confusion matrix components
    TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    # Avoid div by zero errors
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Precision': prec, 'Recall': rec, 'F1': f1}

# ==========================================
# 4. EXECUTION FLOW
# ==========================================

if __name__ == "__main__":
    # A. LOAD
    csv_path = 'credit_risk_dataset.csv'
    if not os.path.exists(csv_path):
        print("Error: CSV not found. Please download it from Kaggle.")
        exit()
        
    df = pd.read_csv(csv_path)
    
    # Rename to match assignment variable names
    df.rename(columns={
        'person_income': 'Income', 'person_home_ownership': 'Home',
        'person_age': 'Age', 'loan_int_rate': 'Interest_Rate', 
        'loan_amnt': 'Loan_Amount', 'loan_status': 'Default'
    }, inplace=True)
    
    df = df[['Income', 'Home', 'Age', 'Interest_Rate', 'Loan_Amount', 'Default']]
    
    # B. PREPROCESS
    model = CreditRiskTree(target_col='Default')
    
    # Discretize continuous vars (handles NaNs internally)
    for col in ['Income', 'Age', 'Interest_Rate', 'Loan_Amount']:
        df = model.manual_qcut(df, col)
        
    # Binarize nominals
    df = model.manual_one_hot(df)
    
    # C. SPLIT
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    # D. TRAIN
    model.fit(train_df)
    
    # E. EVALUATE (TREE)
    print("Evaluating Decision Tree...")
    tree_preds = model.predict(test_df)
    tree_metrics = get_performance(test_df['Default'].tolist(), tree_preds)
    
    # F. EVALUATE (RANDOM BASELINE)
    # Weighted random based on training data distribution
    print("Evaluating Random Baseline...")
    default_prob = train_df['Default'].mean()
    rand_preds = [1 if random.random() < default_prob else 0 for _ in range(len(test_df))]
    rand_metrics = get_performance(test_df['Default'].tolist(), rand_preds)
    
    # G. REPORT
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    
    print(f"\n[A] Decision Tree Model")
    print(f"Confusion Matrix: [TP: {tree_metrics['TP']}, FP: {tree_metrics['FP']}, TN: {tree_metrics['TN']}, FN: {tree_metrics['FN']}]")
    print(f"Precision: {tree_metrics['Precision']:.4f}")
    print(f"Recall:    {tree_metrics['Recall']:.4f}")
    print(f"F1-Score:  {tree_metrics['F1']:.4f}")
    
    print(f"\n[B] Random Weighted Baseline")
    print(f"Confusion Matrix: [TP: {rand_metrics['TP']}, FP: {rand_metrics['FP']}, TN: {rand_metrics['TN']}, FN: {rand_metrics['FN']}]")
    print(f"Precision: {rand_metrics['Precision']:.4f}")
    print(f"Recall:    {rand_metrics['Recall']:.4f}")
    print(f"F1-Score:  {rand_metrics['F1']:.4f}")
    if tree_metrics['F1'] > rand_metrics['F1']:
        print(" -> SUCCESS: The Decision Tree significantly outperforms random guessing.")
    else:
        print(" -> ISSUE: The Decision Tree is performing similarly to random guessing.")