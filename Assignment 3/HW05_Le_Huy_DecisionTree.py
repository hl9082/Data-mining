# =============================================================================
# Program Name: HW_05_Le_Huy_DecisionTree_Trainer.py
# Author: Huy Le
# Purpose: This script ingests training data for "Abominable SnowFolk", 
#          pre-quantizes the continuous attributes to the nearest 2 values,
#          trains a Decision Tree, and dynamically GENERATES a standalone 
#          Classifier program using metaprogramming.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

def main():
    # =========================================================================
    # STEP 1: Data Ingestion
    # =========================================================================
    train_file = 'Data/Abominable_Data_HW_LABELED_TRAINING_DATA__v780_2255.csv'

    print(f"Loading training data from {train_file}...")
    try:
        train_df = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find {train_file}.")
        return

    # Separate features from the target labels
    X_train_raw = train_df.drop(columns=['ClassName', 'ClassID'])
    y_train = train_df['ClassID']
    feature_names = list(X_train_raw.columns)

    # =========================================================================
    # STEP 2: Pre-Quantization
    # Rounding to the nearest multiple of 2 (round(value / 2.0) * 2.0)
    # =========================================================================
    print("Pre-quantizing continuous attributes to the nearest multiple of 2...")
    X_train = (X_train_raw / 2.0).round() * 2.0

    # =========================================================================
    # STEP 3: Train the Decision Tree Classifier
    # =========================================================================
    print("Training the Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    max_depth = clf.tree_.max_depth
    print(f"\n-> MAXIMUM CALL DEPTH USED: {max_depth}")

    # =========================================================================
    # STEP 4: Evaluation and Confusion Matrix
    # =========================================================================
    y_pred_train = clf.predict(X_train)
    cm = confusion_matrix(y_train, y_pred_train, labels=[-1, 1])
    
    cm_df = pd.DataFrame(cm, 
                         index=['Actual: Assam (-1)', 'Actual: Bhuttan (+1)'], 
                         columns=['Predicted: Assam (-1)', 'Predicted: Bhuttan (+1)'])

    print("\n" + "="*60)
    print("CONFUSION MATRIX (TRAINING DATA)")
    print("="*60)
    print(cm_df)
    
    acc = accuracy_score(y_train, y_pred_train)
    print(f"\nFinal Training Accuracy: {acc * 100:.2f}%\n")

    # =========================================================================
    # STEP 5: METAPROGRAMMING - Generate the Classifier Script
    # We write a recursive function to traverse the trained tree and convert 
    # its nodes into Python if/else code as strings.
    # =========================================================================
    print("Generating standalone classifier program: HW_05_Classifier_Le_Huy.py...")
    
    def extract_tree_logic(node, depth):
        indent = "    " * depth
        # If the node is not a leaf node (-2 is scikit-learn's leaf indicator)
        if clf.tree_.feature[node] != -2:
            # Get the attribute name and the threshold value
            name = feature_names[clf.tree_.feature[node]]
            threshold = clf.tree_.threshold[node]
            
            # Note: Scikit-learn uses <= for left children!
            code = f"{indent}if {name} <= {threshold}:\n"
            code += extract_tree_logic(clf.tree_.children_left[node], depth + 1)
            code += f"{indent}else:\n"
            code += extract_tree_logic(clf.tree_.children_right[node], depth + 1)
            return code
        else:
            # If it IS a leaf node, find out which class won the vote
            class_index = np.argmax(clf.tree_.value[node])
            predicted_class = clf.classes_[class_index]
            
            # Add a comment so the grader knows what class this is
            class_name = "Assam" if predicted_class == -1 else "Bhuttan"
            return f"{indent}predicted_class = {predicted_class} # {class_name}\n"

    # Start the recursion at the root node (0) with an initial indentation of 4 tabs
    decision_logic = extract_tree_logic(0, 4)

    # Now, we construct the full string of the new python file
    classifier_code = f"""# =============================================================================
# Program Name: HW_05_Classifier_Le_Huy.py
# Purpose: Auto-generated Decision Tree classifier for Abominable SnowFolk.
# =============================================================================
import csv
import sys

def classify_data(filename):
    print(f"Reading and classifying data from: {{filename}}\\n")
    
    try:
        with open(filename, mode='r') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                # Pre-quantize the inputs exactly the way the Trainer did!
                Age = round(float(row['Age']) / 2.0) * 2.0
                Ht = round(float(row['Ht']) / 2.0) * 2.0
                TailLn = round(float(row['TailLn']) / 2.0) * 2.0
                HairLn = round(float(row['HairLn']) / 2.0) * 2.0
                HairClr = round(float(row['HairClr']) / 2.0) * 2.0
                BangLn = round(float(row['BangLn']) / 2.0) * 2.0
                Reach = round(float(row['Reach']) / 2.0) * 2.0
                EarLobes = round(float(row['EarLobes']) / 2.0) * 2.0

                # --- AUTO-GENERATED DECISION TREE LOGIC ---
{decision_logic}
                # Print out the class value for each line of the test data file
                print(predicted_class)
                
    except FileNotFoundError:
        print(f"ERROR: File '{{filename}}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    # Expect one parameter: the string containing the filename to read in
    if len(sys.argv) < 2:
        print("Usage: python HW_05_Classifier_Le_Huy.py <ValidationData.csv>")
    else:
        classify_data(sys.argv[1])
"""

    # Write out the new python file
    with open('HW_05_Classifier_Le_Huy.py', 'w') as f:
        f.write(classifier_code)
        
    print("SUCCESS: Standalone classifier created!")

if __name__ == "__main__":
    main()