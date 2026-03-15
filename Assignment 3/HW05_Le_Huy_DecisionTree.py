# =============================================================================
# Program Name: HW05_Le_Huy_DecisionTree.py
# Author: Huy Le
# Purpose: This script ingests training data for "Abominable SnowFolk", 
#          pre-quantizes the continuous attributes to the nearest 2 values 
#          (to reduce threshold search space), trains a Decision Tree classifier, 
#          generates a confusion matrix, and predicts on unseen validation data.
# =============================================================================

import pandas as pd # Used for efficient data manipulation and tabular operations
import numpy as np # Used for mathematical rounding operations
from sklearn.tree import DecisionTreeClassifier, export_text # The core Decision Tree model
from sklearn.metrics import confusion_matrix, accuracy_score # Used for evaluation metrics

def main():
    # =========================================================================
    # STEP 1: Data Ingestion
    # We load the training and validation data using pandas.
    # =========================================================================
    train_file = 'Data/Abominable_Data_HW_LABELED_TRAINING_DATA__v780_2255.csv'
    val_file = 'Data/Abominable_VALIDATION_Data_FOR_STUDENTS_v780_2255.csv'

    print(f"Loading training data from {train_file}...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Separate features (X) from the target labels (y) in the training data
    # ClassID is our target: -1 (Assam) or +1 (Bhuttan)
    X_train_raw = train_df.drop(columns=['ClassName', 'ClassID'])
    y_train = train_df['ClassID']

    # For validation data, we don't have the labels yet
    X_val_raw = val_df.copy()

    # =========================================================================
    # STEP 2: Pre-Quantization (Crucial for the Rubric!)
    # Dr. Kinsman requires rounding to the nearest 2 values. 
    # Formula: round(value / 2.0) * 2.0
    # This prevents the algorithm from checking thousands of useless thresholds.
    # =========================================================================
    print("Pre-quantizing continuous attributes to the nearest multiple of 2...")
    
    # We apply the quantization logic to both training and validation features
    # Note: We do this to all columns. If 'EarLobes' or 'HairClr' are categorical, 
    # rounding them to the nearest 2 might alter their categorical meaning, but 
    # standard practice for this specific prompt instruction is global continuous rounding.
    X_train = (X_train_raw / 2.0).round() * 2.0
    X_val = (X_val_raw / 2.0).round() * 2.0

    # =========================================================================
    # STEP 3: Train the Decision Tree Classifier
    # We instantiate and fit the tree. We let the tree grow naturally to see 
    # its maximum call depth, but you can set max_depth=X if needed.
    # =========================================================================
    print("Training the Decision Tree Classifier...")
    
    # Instantiate the classifier. random_state=42 ensures reproducibility.
    clf = DecisionTreeClassifier(random_state=42)
    
    # Fit the model to our quantized training data
    clf.fit(X_train, y_train)

    # Calculate and print the maximum call depth (number of levels)
    # The tree_.max_depth attribute tells us how deep the recursion went
    max_depth = clf.tree_.max_depth
    print(f"\n-> MAXIMUM CALL DEPTH USED: {max_depth}")

    # =========================================================================
    # STEP 4: Evaluation and Confusion Matrix
    # We test the trained model on the same data it learned from to report 
    # the training accuracy and confusion matrix for the write-up.
    # =========================================================================
    y_pred_train = clf.predict(X_train)
    
    # Generate the confusion matrix
    # Labels are ordered: -1 (Assam), 1 (Bhuttan)
    cm = confusion_matrix(y_train, y_pred_train, labels=[-1, 1])
    
    # Format the matrix nicely using pandas for console output
    row_labels = ['Actual: Assam (-1)', 'Actual: Bhuttan (+1)']
    col_labels = ['Predicted: Assam (-1)', 'Predicted: Bhuttan (+1)']
    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)

    print("\n" + "="*60)
    print("CONFUSION MATRIX (TRAINING DATA)")
    print("="*60)
    print(cm_df)

    # Calculate overall accuracy
    acc = accuracy_score(y_train, y_pred_train)
    print(f"\nFinal Training Accuracy: {acc * 100:.2f}%")
    print("="*60 + "\n")

    # (Optional) Print out the textual representation of the tree decisions
    # This helps answer the "What were the most important attributes?" question
    print("DECISION TREE LOGIC:")
    tree_rules = export_text(clf, feature_names=list(X_train.columns))
    print(tree_rules)

    # =========================================================================
    # STEP 5: Generate Validation Classifications
    # Predict the ClassIDs for the validation set and save to a new CSV.
    # =========================================================================
    print("\nClassifying unseen validation data...")
    val_predictions = clf.predict(X_val)
    
    # Add the predictions as a new column to the ORIGINAL validation dataframe
    val_df_output = val_df.copy()
    val_df_output['Guessed_ClassID'] = val_predictions
    
    # Map the IDs back to the Class Names for clarity
    class_map = {-1: 'Assam', 1: 'Bhuttan'}
    val_df_output['Guessed_ClassName'] = val_df_output['Guessed_ClassID'].map(class_map)

    # Save to a new CSV file
    output_filename = 'HW05_Le_Huy_MyClassifications.csv'
    val_df_output.to_csv(output_filename, index=False)
    
    print(f"Successfully saved validation predictions to: {output_filename}")

if __name__ == "__main__":
    main()