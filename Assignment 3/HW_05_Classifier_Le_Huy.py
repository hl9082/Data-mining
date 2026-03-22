# =============================================================================
# Program Name: HW_05_Classifier_Le_Huy.py
# Author: Huy Le (hl9082)
# Purpose: Auto-generated Decision Tree classifier for Abominable SnowFolk.
# =============================================================================
import csv
import sys

def classify_data(filename):
    """
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
    """
    print(f"Reading and classifying data from: {filename}\n")
    
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
                if BangLn <= 7.0:
                    if BangLn <= 5.0:
                        if HairLn <= 13.0:
                            if HairLn <= 11.0:
                                if TailLn <= 9.0:
                                    if Age <= 49.0:
                                        if HairClr <= 1.0:
                                            if Age <= 35.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                        else:
                                            if HairLn <= 9.0:
                                                if Ht <= 147.0:
                                                    if Age <= 45.0:
                                                        if TailLn <= 5.0:
                                                            if Ht <= 132.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Age <= 35.0:
                                                                if Age <= 33.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        if EarLobes <= 0.5:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Ht <= 145.0:
                                                                if HairLn <= 7.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                            else:
                                                                if HairClr <= 11.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                if Age <= 34.0:
                                                    if Age <= 30.0:
                                                        if HairClr <= 6.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    if EarLobes <= 0.5:
                                                        if Age <= 45.0:
                                                            if Age <= 41.0:
                                                                if Ht <= 146.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                    else:
                                        if Reach <= 124.0:
                                            if HairClr <= 9.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                        else:
                                            if TailLn <= 5.0:
                                                if EarLobes <= 0.5:
                                                    if Age <= 59.0:
                                                        if HairClr <= 7.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                predicted_class = -1 # Assam
                                else:
                                    if Ht <= 137.0:
                                        if Age <= 27.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if TailLn <= 13.0:
                                                if HairClr <= 5.0:
                                                    if Ht <= 131.0:
                                                        if Reach <= 135.0:
                                                            if Ht <= 127.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Reach <= 133.0:
                                                                    if HairClr <= 3.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                        else:
                                                            if Age <= 51.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    if Age <= 65.0:
                                                        if EarLobes <= 0.5:
                                                            if Ht <= 129.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            if Ht <= 135.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if HairLn <= 8.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if HairClr <= 9.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                            else:
                                                if Age <= 31.0:
                                                    if EarLobes <= 0.5:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    if Reach <= 141.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        if HairClr <= 11.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Age <= 38.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                    else:
                                        if Age <= 39.0:
                                            if Age <= 37.0:
                                                if TailLn <= 15.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if HairClr <= 10.0:
                                                        if HairClr <= 5.0:
                                                            if Ht <= 151.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Age <= 33.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if TailLn <= 17.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                            else:
                                                if HairLn <= 9.0:
                                                    if TailLn <= 17.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    if Ht <= 145.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                        else:
                                            if Age <= 61.0:
                                                if Age <= 41.0:
                                                    if EarLobes <= 0.5:
                                                        if Ht <= 145.0:
                                                            if Reach <= 149.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if HairLn <= 9.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    if Reach <= 165.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        if Ht <= 161.0:
                                                            if TailLn <= 15.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Age <= 49.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                            else:
                                                if Reach <= 149.0:
                                                    if Age <= 63.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if HairClr <= 4.0:
                                                            if TailLn <= 14.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = -1 # Assam
                            else:
                                if HairClr <= 11.0:
                                    if HairClr <= 3.0:
                                        if Age <= 42.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            predicted_class = -1 # Assam
                                    else:
                                        if HairClr <= 7.0:
                                            predicted_class = -1 # Assam
                                        else:
                                            if Ht <= 151.0:
                                                if Age <= 43.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if TailLn <= 9.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = -1 # Assam
                                else:
                                    if TailLn <= 11.0:
                                        if Ht <= 149.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if Ht <= 153.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                    else:
                                        if Reach <= 128.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            predicted_class = -1 # Assam
                        else:
                            predicted_class = 1 # Bhuttan
                    else:
                        if HairLn <= 9.0:
                            if Age <= 47.0:
                                if TailLn <= 9.0:
                                    if Ht <= 143.0:
                                        if Reach <= 137.0:
                                            if HairClr <= 5.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if Reach <= 131.0:
                                                    if Ht <= 121.0:
                                                        if Reach <= 121.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                        else:
                                            if Age <= 45.0:
                                                if Age <= 37.0:
                                                    if Age <= 35.0:
                                                        if Reach <= 146.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if EarLobes <= 0.5:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                            else:
                                                if Ht <= 139.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                    else:
                                        if Reach <= 149.0:
                                            predicted_class = -1 # Assam
                                        else:
                                            if Age <= 45.0:
                                                if EarLobes <= 0.5:
                                                    if HairClr <= 7.0:
                                                        if Ht <= 148.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Ht <= 151.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairClr <= 5.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                    else:
                                                        if HairLn <= 7.0:
                                                            if HairLn <= 5.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                else:
                                                    if Ht <= 145.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Reach <= 157.0:
                                                            if Age <= 39.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if HairClr <= 5.0:
                                                                    if Reach <= 155.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Reach <= 166.0:
                                                                if Age <= 43.0:
                                                                    if Age <= 31.0:
                                                                        if Ht <= 155.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if HairClr <= 7.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if TailLn <= 7.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                            else:
                                                                if Reach <= 171.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if Age <= 33.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                            else:
                                                if Ht <= 155.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 167.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                else:
                                    if EarLobes <= 0.5:
                                        if Ht <= 157.0:
                                            if Reach <= 149.0:
                                                if Ht <= 128.0:
                                                    if Age <= 44.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    if HairClr <= 9.0:
                                                        if Reach <= 141.0:
                                                            if TailLn <= 13.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Ht <= 133.0:
                                                                    if HairLn <= 7.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                        else:
                                                            if Reach <= 147.0:
                                                                if TailLn <= 13.0:
                                                                    if Age <= 40.0:
                                                                        if Reach <= 145.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            if HairClr <= 7.0:
                                                                                if TailLn <= 11.0:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Reach <= 143.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if Reach <= 145.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                            else:
                                                if Ht <= 147.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Age <= 37.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        if HairClr <= 9.0:
                                                            if Reach <= 153.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            if TailLn <= 11.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                        else:
                                            if Ht <= 161.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if Ht <= 163.0:
                                                    if HairClr <= 7.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = -1 # Assam
                                    else:
                                        if Ht <= 155.0:
                                            if Reach <= 159.0:
                                                if Age <= 37.0:
                                                    if HairLn <= 7.0:
                                                        if Reach <= 147.0:
                                                            if Ht <= 140.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if Ht <= 147.0:
                                                            if Ht <= 145.0:
                                                                if Reach <= 143.0:
                                                                    if HairClr <= 7.0:
                                                                        if Ht <= 134.0:
                                                                            if HairClr <= 2.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            if TailLn <= 11.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    if Age <= 29.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if HairClr <= 7.0:
                                                                            if Reach <= 147.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            if Age <= 23.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if TailLn <= 13.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Reach <= 154.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                else:
                                                    if TailLn <= 17.0:
                                                        if Ht <= 123.0:
                                                            if Ht <= 117.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Reach <= 133.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Ht <= 139.0:
                                                                    if Reach <= 143.0:
                                                                        if TailLn <= 15.0:
                                                                            if HairClr <= 5.0:
                                                                                if HairLn <= 5.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    if TailLn <= 13.0:
                                                                                        if HairClr <= 3.0:
                                                                                            predicted_class = 1 # Bhuttan
                                                                                        else:
                                                                                            if TailLn <= 11.0:
                                                                                                predicted_class = -1 # Assam
                                                                                            else:
                                                                                                predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        predicted_class = -1 # Assam
                                                                            else:
                                                                                if Ht <= 129.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if TailLn <= 15.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            if Age <= 40.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                else:
                                                                    if Age <= 45.0:
                                                                        if HairClr <= 7.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            if HairClr <= 11.0:
                                                                                if Ht <= 151.0:
                                                                                    if Reach <= 153.0:
                                                                                        if TailLn <= 13.0:
                                                                                            predicted_class = -1 # Assam
                                                                                        else:
                                                                                            predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                    else:
                                                                        if HairClr <= 5.0:
                                                                            if HairClr <= 3.0:
                                                                                if Ht <= 145.0:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                        else:
                                                                            if HairClr <= 11.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                        else:
                                            if HairClr <= 1.0:
                                                if Age <= 39.0:
                                                    if Ht <= 159.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                if HairClr <= 9.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if Reach <= 165.0:
                                                        if Ht <= 157.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                            else:
                                if Reach <= 123.0:
                                    if Ht <= 115.0:
                                        if EarLobes <= 0.5:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            predicted_class = -1 # Assam
                                    else:
                                        predicted_class = 1 # Bhuttan
                                else:
                                    if TailLn <= 9.0:
                                        if Ht <= 151.0:
                                            if Reach <= 155.0:
                                                if HairClr <= 1.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if HairClr <= 3.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if HairClr <= 7.0:
                                                            if Ht <= 138.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Reach <= 149.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if TailLn <= 7.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if EarLobes <= 0.5:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Age <= 62.0:
                                                                if Reach <= 130.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Age <= 51.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if HairClr <= 9.0:
                                                                            if Age <= 54.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                if Age <= 57.0:
                                                                                    if Reach <= 145.0:
                                                                                        predicted_class = -1 # Assam
                                                                                    else:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    predicted_class = -1 # Assam
                                                                        else:
                                                                            if Age <= 53.0:
                                                                                if Reach <= 142.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    if TailLn <= 5.0:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        if EarLobes <= 0.5:
                                                                                            predicted_class = -1 # Assam
                                                                                        else:
                                                                                            if TailLn <= 7.0:
                                                                                                predicted_class = -1 # Assam
                                                                                            else:
                                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = 1 # Bhuttan
                                        else:
                                            if HairClr <= 11.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if TailLn <= 7.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                    else:
                                        if Age <= 53.0:
                                            if HairClr <= 5.0:
                                                if TailLn <= 13.0:
                                                    if Reach <= 152.0:
                                                        if Ht <= 119.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Reach <= 131.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairLn <= 7.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if Reach <= 147.0:
                                                                        if TailLn <= 11.0:
                                                                            if Reach <= 143.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                if Ht <= 141.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    if EarLobes <= 0.5:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if Reach <= 149.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            if Ht <= 145.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                    else:
                                                        if Age <= 49.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                else:
                                                    if Ht <= 144.0:
                                                        if Reach <= 147.0:
                                                            if Ht <= 137.0:
                                                                if Reach <= 135.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if TailLn <= 15.0:
                                                                        if EarLobes <= 0.5:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                            else:
                                                if HairLn <= 5.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Ht <= 169.0:
                                                        if Ht <= 131.0:
                                                            if EarLobes <= 0.5:
                                                                if Ht <= 122.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            if Age <= 49.0:
                                                                if Ht <= 161.0:
                                                                    if Ht <= 135.0:
                                                                        if HairClr <= 9.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if Ht <= 151.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            if Reach <= 160.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                if TailLn <= 13.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if Age <= 51.0:
                                                                        if HairClr <= 10.0:
                                                                            if HairLn <= 7.0:
                                                                                if EarLobes <= 0.5:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                        else:
                                            if EarLobes <= 0.5:
                                                if Age <= 57.0:
                                                    if Reach <= 171.0:
                                                        if TailLn <= 11.0:
                                                            if Ht <= 152.0:
                                                                if Reach <= 145.0:
                                                                    if Ht <= 135.0:
                                                                        if Ht <= 127.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            if Ht <= 124.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                if Ht <= 141.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if Ht <= 145.0:
                                                        if TailLn <= 15.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if HairClr <= 6.0:
                                                                if Age <= 57.0:
                                                                    if HairClr <= 2.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                    else:
                                                        if HairClr <= 9.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Age <= 59.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if TailLn <= 11.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    if Reach <= 156.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                        else:
                            if Age <= 53.0:
                                if HairLn <= 11.0:
                                    if Ht <= 147.0:
                                        if Age <= 47.0:
                                            if TailLn <= 9.0:
                                                if HairClr <= 7.0:
                                                    if TailLn <= 7.0:
                                                        if HairClr <= 5.0:
                                                            if TailLn <= 3.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if Age <= 33.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Reach <= 149.0:
                                                                if Reach <= 141.0:
                                                                    if HairClr <= 1.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if EarLobes <= 0.5:
                                                                            if HairClr <= 3.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                            else:
                                                if Ht <= 115.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if TailLn <= 19.0:
                                                        if Age <= 29.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Age <= 31.0:
                                                                if EarLobes <= 0.5:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                            else:
                                                                if Age <= 45.0:
                                                                    if Reach <= 133.0:
                                                                        if Ht <= 127.0:
                                                                            if HairClr <= 9.0:
                                                                                if TailLn <= 11.0:
                                                                                    if EarLobes <= 0.5:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        predicted_class = -1 # Assam
                                                                                else:
                                                                                    if HairClr <= 7.0:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        if TailLn <= 16.0:
                                                                                            predicted_class = -1 # Assam
                                                                                        else:
                                                                                            predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        if HairClr <= 1.0:
                                                                            if Age <= 40.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            if EarLobes <= 0.5:
                                                                                if Reach <= 145.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    if TailLn <= 11.0:
                                                                                        predicted_class = -1 # Assam
                                                                                    else:
                                                                                        if Ht <= 141.0:
                                                                                            if Age <= 39.0:
                                                                                                predicted_class = -1 # Assam
                                                                                            else:
                                                                                                predicted_class = 1 # Bhuttan
                                                                                        else:
                                                                                            predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                if HairClr <= 3.0:
                                                                                    predicted_class = 1 # Bhuttan
                                                                                else:
                                                                                    if HairClr <= 7.0:
                                                                                        if TailLn <= 13.0:
                                                                                            if Ht <= 145.0:
                                                                                                if Ht <= 130.0:
                                                                                                    if TailLn <= 11.0:
                                                                                                        predicted_class = 1 # Bhuttan
                                                                                                    else:
                                                                                                        predicted_class = -1 # Assam
                                                                                                else:
                                                                                                    predicted_class = 1 # Bhuttan
                                                                                            else:
                                                                                                predicted_class = -1 # Assam
                                                                                        else:
                                                                                            if Reach <= 143.0:
                                                                                                predicted_class = -1 # Assam
                                                                                            else:
                                                                                                if Age <= 41.0:
                                                                                                    predicted_class = 1 # Bhuttan
                                                                                                else:
                                                                                                    predicted_class = -1 # Assam
                                                                                    else:
                                                                                        if TailLn <= 13.0:
                                                                                            if Ht <= 131.0:
                                                                                                predicted_class = 1 # Bhuttan
                                                                                            else:
                                                                                                if Ht <= 133.0:
                                                                                                    predicted_class = -1 # Assam
                                                                                                else:
                                                                                                    if TailLn <= 11.0:
                                                                                                        if HairClr <= 10.0:
                                                                                                            predicted_class = 1 # Bhuttan
                                                                                                        else:
                                                                                                            if Ht <= 139.0:
                                                                                                                predicted_class = -1 # Assam
                                                                                                            else:
                                                                                                                predicted_class = 1 # Bhuttan
                                                                                                    else:
                                                                                                        if Reach <= 149.0:
                                                                                                            if HairClr <= 11.0:
                                                                                                                predicted_class = -1 # Assam
                                                                                                            else:
                                                                                                                if Ht <= 139.0:
                                                                                                                    predicted_class = 1 # Bhuttan
                                                                                                                else:
                                                                                                                    predicted_class = -1 # Assam
                                                                                                        else:
                                                                                                            predicted_class = 1 # Bhuttan
                                                                                        else:
                                                                                            predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if TailLn <= 11.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if TailLn <= 15.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = -1 # Assam
                                        else:
                                            if TailLn <= 13.0:
                                                if TailLn <= 7.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 129.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Ht <= 143.0:
                                                            if HairClr <= 5.0:
                                                                if Age <= 51.0:
                                                                    if Reach <= 143.0:
                                                                        if Ht <= 128.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairClr <= 11.0:
                                                                    if Reach <= 147.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if Age <= 51.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                else:
                                                                    if TailLn <= 9.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                        else:
                                                            if EarLobes <= 0.5:
                                                                if Reach <= 151.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                            else:
                                                if HairClr <= 9.0:
                                                    if HairClr <= 7.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        if Reach <= 147.0:
                                                            if EarLobes <= 0.5:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                    else:
                                        if Reach <= 153.0:
                                            if Age <= 28.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if HairClr <= 1.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if TailLn <= 11.0:
                                                        if EarLobes <= 0.5:
                                                            if HairClr <= 3.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                        else:
                                            if Ht <= 155.0:
                                                if Age <= 49.0:
                                                    if Age <= 33.0:
                                                        if Ht <= 151.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Reach <= 157.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if TailLn <= 9.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Age <= 45.0:
                                                            if TailLn <= 11.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairClr <= 5.0:
                                                                    if Age <= 37.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if Reach <= 159.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Age <= 47.0:
                                                                if Reach <= 157.0:
                                                                    if Ht <= 149.0:
                                                                        if TailLn <= 10.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    if EarLobes <= 0.5:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                else:
                                                    if HairClr <= 10.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                            else:
                                                if Age <= 30.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if EarLobes <= 0.5:
                                                        if HairClr <= 3.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if TailLn <= 7.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Age <= 36.0:
                                                                    if Ht <= 167.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                    else:
                                                        if TailLn <= 9.0:
                                                            if HairClr <= 5.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if TailLn <= 7.0:
                                                                    if Reach <= 165.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if Reach <= 172.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                else:
                                                                    if Age <= 43.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if Age <= 47.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Reach <= 165.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                if Ht <= 167.0:
                                                                    if TailLn <= 11.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if Age <= 39.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            if HairClr <= 2.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                if HairClr <= 10.0:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Reach <= 183.0:
                                                                        if TailLn <= 13.0:
                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            if Reach <= 174.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                else:
                                    if Age <= 49.0:
                                        if TailLn <= 7.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if HairClr <= 11.0:
                                                if Ht <= 149.0:
                                                    if Reach <= 133.0:
                                                        if Ht <= 127.0:
                                                            if HairLn <= 13.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairClr <= 5.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Age <= 47.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if EarLobes <= 0.5:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if HairLn <= 13.0:
                                                                if Age <= 31.0:
                                                                    if Ht <= 141.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    if HairClr <= 1.0:
                                                                        if TailLn <= 13.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        if Ht <= 139.0:
                                                                            if Ht <= 131.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                if Reach <= 138.0:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    if Age <= 43.0:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        if Ht <= 136.0:
                                                                                            predicted_class = 1 # Bhuttan
                                                                                        else:
                                                                                            predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 155.0:
                                                        if TailLn <= 10.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Reach <= 173.0:
                                                            if Ht <= 163.0:
                                                                if Ht <= 153.0:
                                                                    if TailLn <= 15.0:
                                                                        if Reach <= 157.0:
                                                                            if HairClr <= 3.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                if EarLobes <= 0.5:
                                                                    if TailLn <= 12.0:
                                                                        if Reach <= 171.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = 1 # Bhuttan
                                    else:
                                        if EarLobes <= 0.5:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if Ht <= 139.0:
                                                if TailLn <= 6.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if TailLn <= 13.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Age <= 51.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if HairClr <= 6.0:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                            else:
                                                if TailLn <= 11.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                            else:
                                if EarLobes <= 0.5:
                                    if Reach <= 163.0:
                                        if Reach <= 142.0:
                                            if Ht <= 127.0:
                                                predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = -1 # Assam
                                        else:
                                            if HairClr <= 3.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if HairClr <= 11.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 151.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                    else:
                                        predicted_class = -1 # Assam
                                else:
                                    if HairClr <= 7.0:
                                        if TailLn <= 9.0:
                                            if Reach <= 157.0:
                                                predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = -1 # Assam
                                        else:
                                            if HairLn <= 11.0:
                                                if Ht <= 138.0:
                                                    if Age <= 65.0:
                                                        if HairClr <= 3.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                if Reach <= 123.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                    else:
                                        if Age <= 55.0:
                                            if TailLn <= 11.0:
                                                if Reach <= 155.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                predicted_class = -1 # Assam
                                        else:
                                            if HairClr <= 11.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if Ht <= 147.0:
                                                    predicted_class = -1 # Assam
                                                else:
                                                    if Ht <= 151.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                else:
                    if HairLn <= 9.0:
                        if Age <= 55.0:
                            if TailLn <= 19.0:
                                if HairClr <= 3.0:
                                    if HairLn <= 5.0:
                                        predicted_class = -1 # Assam
                                    else:
                                        if TailLn <= 15.0:
                                            if Age <= 47.0:
                                                if TailLn <= 9.0:
                                                    if Ht <= 147.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Ht <= 149.0:
                                                            if TailLn <= 7.0:
                                                                if Age <= 37.0:
                                                                    predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = 1 # Bhuttan
                                            else:
                                                if TailLn <= 11.0:
                                                    if Ht <= 136.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                        else:
                                            if Ht <= 153.0:
                                                if EarLobes <= 0.5:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Ht <= 137.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = -1 # Assam
                                            else:
                                                predicted_class = -1 # Assam
                                else:
                                    if Reach <= 179.0:
                                        if Age <= 29.0:
                                            if Ht <= 144.0:
                                                if TailLn <= 15.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                            else:
                                                if Age <= 27.0:
                                                    if Reach <= 159.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                        else:
                                            if Age <= 39.0:
                                                if Reach <= 157.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if HairClr <= 9.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Age <= 34.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                            else:
                                                if TailLn <= 9.0:
                                                    if HairClr <= 7.0:
                                                        if TailLn <= 7.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Reach <= 145.0:
                                                                if Age <= 47.0:
                                                                    if Reach <= 143.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        if Age <= 45.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                                else:
                                                    if Age <= 41.0:
                                                        if Ht <= 155.0:
                                                            if Reach <= 149.0:
                                                                if Ht <= 137.0:
                                                                    if HairClr <= 7.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if Age <= 47.0:
                                                            if Reach <= 145.0:
                                                                if Ht <= 135.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if EarLobes <= 0.5:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        if TailLn <= 14.0:
                                                                            if HairClr <= 8.0:
                                                                                predicted_class = -1 # Assam
                                                                            else:
                                                                                if HairLn <= 7.0:
                                                                                    predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            if Ht <= 161.0:
                                                                if TailLn <= 11.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Reach <= 135.0:
                                                                        if Reach <= 132.0:
                                                                            predicted_class = 1 # Bhuttan
                                                                        else:
                                                                            predicted_class = -1 # Assam
                                                                    else:
                                                                        if Age <= 51.0:
                                                                            if Ht <= 143.0:
                                                                                predicted_class = 1 # Bhuttan
                                                                            else:
                                                                                if Age <= 49.0:
                                                                                    if Reach <= 157.0:
                                                                                        predicted_class = 1 # Bhuttan
                                                                                    else:
                                                                                        if EarLobes <= 0.5:
                                                                                            predicted_class = 1 # Bhuttan
                                                                                        else:
                                                                                            predicted_class = -1 # Assam
                                                                                else:
                                                                                    predicted_class = -1 # Assam
                                                                        else:
                                                                            predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                    else:
                                        predicted_class = -1 # Assam
                            else:
                                predicted_class = -1 # Assam
                        else:
                            if EarLobes <= 0.5:
                                predicted_class = -1 # Assam
                            else:
                                if Ht <= 149.0:
                                    if HairClr <= 1.0:
                                        predicted_class = -1 # Assam
                                    else:
                                        if Reach <= 144.0:
                                            if Ht <= 137.0:
                                                predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = -1 # Assam
                                        else:
                                            predicted_class = 1 # Bhuttan
                                else:
                                    predicted_class = -1 # Assam
                    else:
                        if Age <= 47.0:
                            if HairLn <= 11.0:
                                if Reach <= 171.0:
                                    if TailLn <= 17.0:
                                        if Age <= 45.0:
                                            if Age <= 31.0:
                                                if Ht <= 141.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 147.0:
                                                        if TailLn <= 10.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                                    else:
                                                        if HairClr <= 7.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if HairClr <= 9.0:
                                                                if TailLn <= 9.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Ht <= 157.0:
                                                                        predicted_class = -1 # Assam
                                                                    else:
                                                                        predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                            else:
                                                if TailLn <= 7.0:
                                                    if Age <= 43.0:
                                                        if Reach <= 141.0:
                                                            if Ht <= 135.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if EarLobes <= 0.5:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        if HairClr <= 7.0:
                                                            if EarLobes <= 0.5:
                                                                predicted_class = -1 # Assam
                                                            else:
                                                                predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                else:
                                                    if Age <= 35.0:
                                                        if TailLn <= 11.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            if EarLobes <= 0.5:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                if HairClr <= 5.0:
                                                                    predicted_class = 1 # Bhuttan
                                                                else:
                                                                    if Reach <= 142.0:
                                                                        predicted_class = 1 # Bhuttan
                                                                    else:
                                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                        else:
                                            if Reach <= 130.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if Reach <= 149.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if EarLobes <= 0.5:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                    else:
                                        if EarLobes <= 0.5:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if HairClr <= 6.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                else:
                                    if TailLn <= 7.0:
                                        if HairClr <= 9.0:
                                            predicted_class = -1 # Assam
                                        else:
                                            predicted_class = 1 # Bhuttan
                                    else:
                                        predicted_class = 1 # Bhuttan
                            else:
                                if Reach <= 137.0:
                                    if Reach <= 135.0:
                                        predicted_class = 1 # Bhuttan
                                    else:
                                        if Age <= 34.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if Ht <= 131.0:
                                                if HairClr <= 5.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if HairClr <= 9.0:
                                                        predicted_class = -1 # Assam
                                                    else:
                                                        predicted_class = 1 # Bhuttan
                                            else:
                                                predicted_class = -1 # Assam
                                else:
                                    predicted_class = 1 # Bhuttan
                        else:
                            if HairLn <= 11.0:
                                if Reach <= 173.0:
                                    if TailLn <= 13.0:
                                        if Ht <= 143.0:
                                            predicted_class = 1 # Bhuttan
                                        else:
                                            if Ht <= 149.0:
                                                if TailLn <= 7.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if HairClr <= 6.0:
                                                        if TailLn <= 9.0:
                                                            if HairClr <= 3.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                                        else:
                                                            predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Age <= 49.0:
                                                            predicted_class = 1 # Bhuttan
                                                        else:
                                                            predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                    else:
                                        if Age <= 55.0:
                                            if Age <= 51.0:
                                                if EarLobes <= 0.5:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    if Reach <= 140.0:
                                                        predicted_class = 1 # Bhuttan
                                                    else:
                                                        if Ht <= 140.0:
                                                            predicted_class = -1 # Assam
                                                        else:
                                                            if Reach <= 154.0:
                                                                predicted_class = 1 # Bhuttan
                                                            else:
                                                                predicted_class = -1 # Assam
                                            else:
                                                predicted_class = 1 # Bhuttan
                                        else:
                                            if Ht <= 140.0:
                                                predicted_class = -1 # Assam
                                            else:
                                                if Ht <= 148.0:
                                                    predicted_class = 1 # Bhuttan
                                                else:
                                                    predicted_class = -1 # Assam
                                else:
                                    predicted_class = -1 # Assam
                            else:
                                if Ht <= 159.0:
                                    predicted_class = 1 # Bhuttan
                                else:
                                    if Reach <= 165.0:
                                        if Age <= 50.0:
                                            predicted_class = -1 # Assam
                                        else:
                                            predicted_class = 1 # Bhuttan
                                    else:
                                        predicted_class = 1 # Bhuttan

                # Print out the class value for each line of the test data file
                print(predicted_class)
                
    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    # Expect one parameter: the string containing the filename to read in
    if len(sys.argv) < 2:
        print("Usage: python HW_05_Classifier_Le_Huy.py <ValidationData.csv>")
    else:
        classify_data(sys.argv[1])
