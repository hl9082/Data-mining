# =============================================================================
# Program Name: HW05_Le_Huy_DecisionTree.py
# Author: Huy Le
# Purpose: This script ingests training data for "Abominable SnowFolk", 
#          pre-quantizes the continuous attributes to the nearest 2 values,
#          trains a Decision Tree, and dynamically GENERATES a standalone 
#          Classifier program using metaprogramming.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    """
    Executes the main pipeline for the Decision Tree Trainer.

    This function reads the training data, pre-quantizes the continuous features
    to the nearest multiple of 2 (to reduce threshold search space), trains a 
    scikit-learn DecisionTreeClassifier, and evaluates its performance by 
    calculating the max depth, confusion matrix, and overall accuracy. Finally, 
    it dynamically generates a standalone Python classifier script containing the 
    exact decision tree logic via metaprogramming.

    Args:
        None

    Returns:
        None
    """
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
    # We use the pandas .drop() function to completely remove the 
    # target labels from our feature dataset so the model doesn't cheat.
    X_train_raw = train_df.drop(columns=['ClassName', 'ClassID'])
    y_train = train_df['ClassID']
    feature_names = list(X_train_raw.columns)

    # =========================================================================
    # STEP 2: Pre-Quantization
    # Rounding to the nearest multiple of 2 (round(value / 2.0) * 2.0)
    # =========================================================================
    print("Pre-quantizing continuous attributes to the nearest multiple of 2...")

    # Save the original binary/categorical columns so they don't get squished
    earlobes_raw = X_train_raw['EarLobes']

    X_train = (X_train_raw / 2.0).round() * 2.0

    # Restore the un-quantized EarLobes column
    X_train['EarLobes'] = earlobes_raw

    # =========================================================================
    # STEP 3: Train the Decision Tree Classifier
    # =========================================================================
    print("Training the Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # The scikit-learn .fit() method mathematically 
    # builds the decision tree by calculating Gini impurity to find the best thresholds.

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
        """
        Recursively traverses the trained Decision Tree to extract routing logic.

        This function reads the internal arrays of a fitted scikit-learn
        DecisionTreeClassifier (like tree_.feature and tree_.threshold) and 
        generates equivalent Python if/else statements as formatted strings.

        Args:
            node (int): The current node index in the decision tree.
            depth (int): The current indentation depth for formatting the Python code.

        Returns:
            str: A multi-line string containing the Python if/else logic for the
                 current node and all of its recursive children. 
        """
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
            
            # Add a comment so we know what class this is
            class_name = "Assam" if predicted_class == -1 else "Bhuttan"
            return f"{indent}predicted_class = {predicted_class} # {class_name}\n"

    # Start the recursion at the root node (0) with an initial indentation of 4 tabs
    decision_logic = extract_tree_logic(0, 4)

    # Now, we construct the full string of the new python file
    classifier_code = f"""# =============================================================================
# Program Name: HW_05_Classifier_Le_Huy.py
# Author: Huy Le (hl9082)
# Purpose: Auto-generated Decision Tree classifier for Abominable SnowFolk.
# =============================================================================
import csv
import sys

def classify_data(filename):
    \"\"\"
    Reads validation data and classifies it using the hardcoded Decision Tree logic.

    This function opens the specified CSV file, pre-quantizes the input features
    (rounding to the nearest 2.0) exactly as they were pre-processed during training, 
    and applies a series of auto-generated if/else statements to predict the 
    specimen's class. The resulting classification (-1 for Assam, +1 for Bhuttan) 
    is printed directly to standard output for each row.

    Args:
        filename (str): The relative or absolute path to the validation CSV file.

    Returns:
        None: Output is printed to stdout.

    Raises:
        SystemExit: If the provided filename does not exist.
    \"\"\"
    print(f"Reading and classifying data from: {{filename}}\\n")
    
    try:
        with open(filename, mode='r') as file:
        # csv.DictReader reads the file line-by-line and maps each value to its column header name, 
        # allowing us to access data via row['Age']
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
                EarLobes = int(row['EarLobes']) 

                # --- DECISION TREE LOGIC ---
                # 
                # HOW IT WORKS:
                # 1. The Trainer fit a scikit-learn DecisionTreeClassifier to the training data.
                # 2. A recursive function traversed the tree's internal 
                #    arrays.
                # 3. At each node, it extracted the optimal threshold (based on Gini Impurity or 
                #    Information Gain) and dynamically wrote the corresponding Python 'if' statement.
                # 4. If the data failed the threshold, it wrote the 'else' statement.
                # 5. When it reached a pure leaf node, it assigned the predicted_class variable
                #    (-1 for Assam, +1 for Bhuttan).
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

    # Write out the python code for the classifier program
    with open('HW_05_Classifier_Le_Huy.py', 'w') as f:
        f.write(classifier_code)
        
    print("SUCCESS: Standalone classifier created!")

if __name__ == "__main__":
    main()