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
# Rubric Requirement: "Comparison & Discussion" implies consistent results.
# Setting seeds ensures the 'Random Model' and 'Tree Building' are reproducible.
random.seed(42)
np.random.seed(42)

# ==========================================
# 2. DATA LOADING (Via KaggleHub)
# ==========================================
def load_and_clean_data():

    """
    Loads the Credit Risk dataset from the local directory, renames columns to match
    assignment logic, and filters for relevant features.

    @return df - A Pandas DataFrame containing the cleaned and renamed dataset.
    """

    # 1. Define the expected local filename
    csv_file = 'credit_risk_dataset.csv'
    
    # 2. Check if the file actually exists locally
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"ERROR: Could not find '{csv_file}' in the current folder. "
            "Please download the CSV from Kaggle and place it next to this script."
        )
    
    print(f"--- Loading '{csv_file}' from local directory ---")
    df = pd.read_csv(csv_file)
    
    # 3. RENAME columns (Logic remains the same as before)
    # This is crucial so the rest of your assignment code (Hunt's algorithm) 
    # can find 'Income', 'Default', etc.
    rename_map = {
        'person_income': 'Income',
        'person_home_ownership': 'Home',
        'person_age': 'Age',
        'loan_int_rate': 'Interest_Rate', 
        'loan_amnt': 'Loan_Amount',
        'loan_status': 'Default'
    }
    df = df.rename(columns=rename_map)
    
    # 4. Filter columns
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
# 4. TREE CLASSES
# ==========================================

class LeafNode:
    """
    Represents a terminal node in the decision tree that holds a final class prediction.
    """
    def __init__(self, class_label):
        """
        Initializes the LeafNode.
        @param class_label - The predicted class (0 or 1).
        """
        self.class_label = class_label # The prediction (0 or 1)

    def is_leaf(self):
        """
        Checks if the node is a leaf.
        @return True.
        """
        return True

class DecisionNode:
    """
    Represents a split point in the decision tree containing a test condition
    and references to child nodes.
    """
    def __init__(self, split_attribute, majority_class):
        """
        Initializes the DecisionNode.
        @param split_attribute - The name of the attribute used for splitting.
        @param majority_class - The majority class of records at this node (fallback).
        """
        self.split_attribute = split_attribute
        self.children = {} # Dictionary: {value: Node}
        self.majority_class = majority_class # Fallback for unseen/missing values during prediction

    def is_leaf(self):
        """
        Checks if the node is a leaf.
        @return False
        """
        return False

    def add_child(self, value, node):
        """
        Adds a branch to the decision node.
        @param value - The value of the split attribute for this branch.
        @param node - The child node (Leaf or Decision) associated with the value.
        """
        self.children[value] = node

# ==========================================
# 5. HUNT'S ALGORITHM (CORE LOGIC)
# ==========================================

def get_majority_class(records, target_name):
    """
    Finds the most frequent class label in a set of records.
    Rubric Requirement: Tie-breaking handled consistently.

    @param records - The subset of dataframe records.
    @param target_name - The name of the target column.
    @return - The mode (majority class) of the target column (0 or 1).
    """
    if len(records) == 0: 
        return 0
    
    # value_counts() sorts by frequency. If there is a tie, Pandas picks based on index order.
    # We can rely on this for consistency, or add explicit logic.
    return records[target_name].mode()[0]

def calculate_weighted_gini(records, attribute, target_name):
    """
    Calculates the weighted Gini Index for a specific attribute split.
    Rubric Requirement: "Impurity calculations correctly ignore the specific 
    missing attribute for affected records."

    @param records - The subset of dataframe records.
    @param attribute - The attribute to evaluate for splitting.
    @param target_name - The name of the target column.
    @return weighted_gini - The calculated weighted Gini Index (float).
    """
    # 1. Identify Valid Records (Ignore NaNs for this specific attribute)
    valid_mask = records[attribute].notna()
    valid_records = records[valid_mask]
    
    total_valid = len(valid_records)
    
    # If this attribute is missing for EVERYONE, it's a useless split.
    if total_valid == 0:
        return 1.0 # Max impurity

    weighted_gini = 0.0
    unique_values = valid_records[attribute].unique()
    
    for val in unique_values:
        # Subset based on the valid branch
        subset = valid_records[valid_records[attribute] == val]
        subset_size = len(subset)
        
        # Calculate Gini for this child node
        # Formula: 1 - sum(p^2)
        gini_node = 1.0
        classes = subset[target_name].unique()
        for c in classes:
            prob = len(subset[subset[target_name] == c]) / subset_size
            gini_node -= prob**2
            
        # Add to weighted average
        weighted_gini += (subset_size / total_valid) * gini_node
        
    return weighted_gini

def find_best_split(records, attributes, target_name):
    """
    Iterates through all attributes to find the one that minimizes the Weighted Gini Index.
    Rubric: "Select the attribute test condition that maximizes the gain in purity."

    @param records - The subset of dataframe records.
    @param attributes - List of available attribute names.
    @param target_name - The name of the target column.
    @return best_attr - The name of the attribute with the lowest Gini Index.
    @return best_gini - The score of the best attribute (added for printing).
    """
    best_gini = float('inf')
    best_attr = None
    
    for attr in attributes:
        gini = calculate_weighted_gini(records, attr, target_name)
        if gini < best_gini:
            best_gini = gini
            best_attr = attr
            
    return best_attr

def build_tree(records, attributes, target_name, depth=0):
    """
    Recursively builds the decision tree using Hunt's Algorithm.
    
    @param records - The subset of dataframe records for the current node.
    @param attributes - List of available attributes for splitting.
    @param target_name - The name of the target column.
    @param depth - The current depth of the tree (used for printing indentation).
    @return node - The root node of the constructed tree (or subtree).
    """

    best_gini = float('inf')

    # Helper indentation for printing
    indent = "   | " * depth

    # STOPPING CONDITION 1: Purity (All records same class) 
    unique_classes = records[target_name].unique()
    if len(unique_classes) == 1:
        return LeafNode(unique_classes[0])

    # STOPPING CONDITION 2: Fragmentation (Records < 5)
    if len(records) < 5:
        return LeafNode(get_majority_class(records, target_name))

    # STOPPING CONDITION 3: No Attributes Left
    if len(attributes) == 0:
        return LeafNode(get_majority_class(records, target_name))

    # --- RECURSION ---
    best_attr = find_best_split(records, attributes, target_name)
    
    # Edge Case: If no split improves anything (or all Gini=1.0), stop
    if best_attr is None:
        return LeafNode(get_majority_class(records, target_name))

    # PRINTING THE GINI INDEX (Requested Feature)
    print(f"{indent}Split on [{best_attr}] (Weighted Gini: {best_gini:.4f})")

    # Create Decision Node
    node = DecisionNode(best_attr, majority_class=get_majority_class(records, target_name))
    
    # Remove used attribute
    remaining_attrs = [a for a in attributes if a != best_attr]
    
    # Create Children
    # Note: We must handle NaNs here. If a row has NaN for best_attr, it technically
    # cannot go down a specific branch. For training construction, we usually just 
    # iterate over the KNOWN values.
    distinct_values = records[best_attr].dropna().unique()
    
    for val in distinct_values:
        subset = records[records[best_attr] == val]
        
        if len(subset) == 0:
            # If a branch is empty, leaf with parent's majority
            child = LeafNode(get_majority_class(records, target_name))
        else:
            # Pass depth + 1 for correct indentation printing
            child = build_tree(subset, remaining_attrs, target_name, depth+1)
            
        node.add_child(val, child)

    return node

# ==========================================
# 6. EVALUATION & METRICS
# ==========================================

def predict_single(node, row):
    """
    Traverses the decision tree to predict the class for a single row.

    @param node - The current node in the tree traversal.
    @param row - The data row (pandas Series) to predict.
    @return - The predicted class label (0 or 1).
    """
    if node.is_leaf():
        return node.class_label
    
    attr = node.split_attribute
    val = row.get(attr)
    
    # Handle Missing Values or Unseen Categories during Test Time
    # If the value is NaN or we never saw this value in training -> Use Majority Fallback
    if pd.isna(val) or val not in node.children:
        return node.majority_class
    
    return predict_single(node.children[val], row)

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
# 7. MAIN EXECUTION
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
    
    print(f"\n[Step 2] Training Tree on {len(train_df)} records...")
    available_attributes = [c for c in df.columns if c != target_col]
    
    print("--- Start Tree Construction ---")
    
    # Build Tree (Pass depth=0 to start indentation)
    tree_root = build_tree(train_df, available_attributes, target_col, depth=0)
    
    print("--- End Tree Construction ---")
    print(" -> Tree construction complete.")
    
    print("\n[Step 3] Evaluation on Test Set...")
    
    # A. Tree Predictions
    tree_preds = [predict_single(tree_root, row) for _, row in test_df.iterrows()]
    actuals = test_df[target_col].tolist()
    
    tree_metrics = calculate_metrics(actuals, tree_preds)
    
    # B. Random Weighted Baseline
    # "Compare... with a simple random model weighted based on probability"
    prob_default = train_df[target_col].mean()
    print(f" -> Training Data Default Probability: {prob_default:.2%}")
    
    random_preds = [1 if random.random() < prob_default else 0 for _ in range(len(test_df))]
    random_metrics = calculate_metrics(actuals, random_preds)
    
    # C. Display Results
    print("\n========================================")
    print("RESULTS SUMMARY")
    print("========================================")
    
    print("\n[A] My Decision Tree Model")
    print_confusion_matrix(tree_metrics)
    print(f"Precision: {tree_metrics['Precision']:.4f}")
    print(f"Recall:    {tree_metrics['Recall']:.4f}")
    print(f"F1-Score:  {tree_metrics['F1']:.4f}")
    
    print("\n[B] Random Weighted Baseline")
    print_confusion_matrix(random_metrics)
    print(f"Precision: {random_metrics['Precision']:.4f}")
    print(f"Recall:    {random_metrics['Recall']:.4f}")
    print(f"F1-Score:  {random_metrics['F1']:.4f}")
    
    print("\n[C] Discussion Prompt")
    print("Compare the 'Tree F1' vs 'Random F1'.")
    if tree_metrics['F1'] > random_metrics['F1']:
        print(" -> SUCCESS: The Decision Tree significantly outperforms random guessing.")
    else:
        print(" -> ISSUE: The Decision Tree is performing similarly to random guessing.")