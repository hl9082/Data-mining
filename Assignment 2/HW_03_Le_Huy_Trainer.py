# =============================================================================
# Program Name: HW_03_Le_Huy_Trainer.py
# Author: Huy Le (hl9082)
# Class: Data Mining
# Purpose: This script acts as the "Trainer" or "Mentor" program. It reads in 
#          training data, pre-quantizes the speed attribute, tests all possible 
#          speed thresholds to find the lowest misclassification rate, 
#          plots an ROC curve and scatterplot, and automatically generates a 
#          standalone Classifier program.
# =============================================================================

import csv #we use the built-in csv module to read and write CSV files
import glob #we use glob to find all CSV files in the Data folder
import matplotlib.pyplot as plt #we use matplotlib to plot the ROC curve and scatterplot
import os
import random #we use random to shuffle our data before balancing the classes
import pandas as pd  #we use pandas to print out the confusion matrix

def main(): # The main function that orchestrates the training process and generates the classifier.
    # =========================================================================
    # STEP 1: Data Ingestion and Pre-Quantization
    # We will loop through all available CSV files (e.g., your 32 files), 
    # extract the speed and intent, and round the speed to the nearest integer 
    # to reduce noise and computation time.
    # =========================================================================
    
    all_speeds = [] # we store all quantized speeds here
    all_intents = [] #we store all corresponding intents here (1 for non-aggressive, 2 for aggresive)
    
    # 1. Find exactly where this Python script is saved on your hard drive
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Tell Python to change its working directory to that folder
    os.chdir(script_directory)
    
    # 3. Create a safe path to the 'Data' folder and look for CSVs inside it
    # This creates a search pattern like: "Data/*.csv"
    search_pattern = os.path.join('Data', '*.csv')
    list_of_csv_files = glob.glob(search_pattern)
    
    # Debugging print to confirm we found the files
    print(f"DEBUG: Found {len(list_of_csv_files)} CSV files in the 'Data' folder.")
    
    # Safety Check: Stop the program cleanly if no files are found
    if len(list_of_csv_files) == 0:
        print("CRITICAL ERROR: No CSV files were found. Please check your 'Data' folder!")
        return # Exits the program so it doesn't crash on min()
    
    # loop through each CSV file and extract the speed and intent data
    for current_filename in list_of_csv_files:
        with open(current_filename, mode='r') as current_file:
            csv_reader = csv.reader(current_file)
            header_row = next(csv_reader) # Skip the header
            
            # Find the index of the columns we need. 
            # Assuming headers are 'Speed' and 'Intent' (or similar).
            # If your data has fixed columns, you can hardcode the indices.
            speed_column_index = header_row.index('Speed')
            intent_column_index = len(header_row) - 1 # Last attribute is target
            
            for current_row in csv_reader:
                # Convert from string -> float -> rounded integer
                raw_speed_string = current_row[speed_column_index]
                quantized_speed = int(float(raw_speed_string)+0.5)
                
                # Extract the target intent (1 = non-aggressive, 2 = aggressive)
                target_intent = int(current_row[intent_column_index])
                
                all_speeds.append(quantized_speed)
                all_intents.append(target_intent)

    # =========================================================================
    # STEP 1B: Data Balancing (Crucial for the rubric!)
    # Separate the data by intent to balance the classes.
    # =========================================================================
    aggressive_records = [] # we store all aggressive records here
    non_aggressive_records = [] #we store all non-aggressive records here
    
    # Group the speeds by their intent
    for i in range(len(all_speeds)):
        if all_intents[i] == 2:
            aggressive_records.append(all_speeds[i])
        else:
            non_aggressive_records.append(all_speeds[i])
            
    print(f"DEBUG: Before balancing - Aggressive: {len(aggressive_records)}, Non-Aggressive: {len(non_aggressive_records)}")
    
    # Find out which class has fewer records
    minimum_class_size = min(len(aggressive_records), len(non_aggressive_records))
    
    SEED=42 # we set a seed = 42 for reproducibility. This means every time we run the program, the random shuffle will be the same, and we will drop the same records to balanace the classes.
    # Set a random seed so our random shuffle is the exact same every time
    random.seed(SEED)
    # Shuffle the lists so we drop random records, not just the ones at the end
    random.shuffle(aggressive_records)
    random.shuffle(non_aggressive_records)
    
    # Truncate both lists to the size of the smaller class
    balanced_aggressive = aggressive_records[:minimum_class_size]
    balanced_non_aggressive = non_aggressive_records[:minimum_class_size]
    
    # Rebuild our all_speeds and all_intents lists with the balanced data
    all_speeds = balanced_aggressive + balanced_non_aggressive
    # Create a matching list of intents (2s for aggressive, 1s for non-aggressive)
    all_intents = [2] * minimum_class_size + [1] * minimum_class_size
    
    # =========================================================================

    total_records = len(all_speeds) # after balancing 
    # this is the total number of records we have to work with
    print(f"Successfully loaded and quantized {total_records} records.")

    # =========================================================================
    # STEP 2: Find the Best Threshold & Calculate ROC Data
    # Get the min and max speeds. We will test every integer between them as a 
    # potential threshold. We track the misclassification rate, True 
    # Positive Rate (TPR), and False Alarm Rate (FAR).
    # =========================================================================
    
    minimum_speed = min(all_speeds) # find the min speed in training data
    maximum_speed = max(all_speeds) # find max speed in training data
    
    lowest_misclassification_rate = 1.0 # Start at 100% error
    best_speed_threshold = minimum_speed # initialize to the lowest threshold possible
    
    # Lists to store metrics for our ROC curve
    true_positive_rates = [] # TPR=TP/(TP+FN)
    false_alarm_rates = [] #FAR=FP/(FP+TN)
    
    # Loop through all possible speeds
    for test_threshold in range(minimum_speed, maximum_speed + 1):
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        
        # Test the current threshold against every record
        for record_index in range(total_records):
            actual_speed = all_speeds[record_index]
            actual_intent = all_intents[record_index]
            
            # Apply our One-Rule logic: speed >= threshold means aggressive (2)
            if actual_speed >= test_threshold:
                guessed_intent = 2
            else:
                guessed_intent = 1
                
            # Tally up our results
            if guessed_intent == 2 and actual_intent == 1:
                false_positives += 1
            elif guessed_intent == 1 and actual_intent == 2:
                false_negatives += 1
            elif guessed_intent == 2 and actual_intent == 2:
                true_positives += 1
            elif guessed_intent == 1 and actual_intent == 1:
                true_negatives += 1
                
        # Calculate the misclassification rate for this threshold
        current_misclassification_rate = (false_positives + false_negatives) / total_records
        
        # If this is the best we've seen, save it!
        if current_misclassification_rate < lowest_misclassification_rate:
            lowest_misclassification_rate = current_misclassification_rate
            best_speed_threshold = test_threshold
            
        # Calculate TPR and FAR for the ROC Curve
        total_actual_aggressives = true_positives + false_negatives
        total_actual_non_aggressives = false_positives + true_negatives
        
        # Avoid division by zero
        if total_actual_aggressives > 0:
            current_tpr = true_positives / total_actual_aggressives
        else:
            current_tpr = 0.0
            
        if total_actual_non_aggressives > 0:
            current_far = false_positives / total_actual_non_aggressives
        else:
            current_far = 0.0
            
        true_positive_rates.append(current_tpr)
        false_alarm_rates.append(current_far)

    print(f"Best Speed Threshold: >= {best_speed_threshold}")
    print(f"Lowest Misclassification Rate: {lowest_misclassification_rate}")


    # =========================================================================
    # STEP 2B: Generate Final Confusion Matrix for the Best Threshold
    # We must report how well our final chosen threshold performed on the 
    # training data for the final write-up.
    # =========================================================================
    
    best_tp = 0
    best_fp = 0
    best_tn = 0
    best_fn = 0
    
    # Loop through the quantized training data one last time to tally the final metrics
    for record_index in range(total_records):
        actual_speed = all_speeds[record_index]
        actual_intent = all_intents[record_index]
        
        # Apply our best discovered threshold rule
        if actual_speed >= best_speed_threshold:
            guessed_intent = 2
        else:
            guessed_intent = 1
            
        # Tally the results into our Confusion Matrix categories
        if guessed_intent == 2 and actual_intent == 1:
            best_fp += 1
        elif guessed_intent == 1 and actual_intent == 2:
            best_fn += 1
        elif guessed_intent == 2 and actual_intent == 2:
            best_tp += 1
        elif guessed_intent == 1 and actual_intent == 1:
            best_tn += 1

    # Create a dictionary organizing your results into columns
    matrix_data = {
        'Predicted: Aggressive (2)': [best_tp, best_fp],
        'Predicted: Non-Aggressive (1)': [best_fn, best_tn]
    }
    
    # Define the row labels
    row_labels = ['Actual: Aggressive (2)', 'Actual: Non-Aggressive (1)']
    
    # Build the Pandas DataFrame to properly format the confusion matrix for display
    confusion_matrix_df = pd.DataFrame(matrix_data, index=row_labels)
    
    print("\n" + "="*60)
    print("FINAL CONFUSION MATRIX FOR WRITE-UP (TRAINING DATA)")
    print("="*60)
    print(confusion_matrix_df)
    
    # Calculate overall accuracy
    final_accuracy = (best_tp + best_tn) / total_records
    print(f"\nFinal Training Accuracy: {final_accuracy * 100:.2f}%")
    print("="*60 + "\n")

    # =========================================================================
    # STEP 3A: Plotting the ROC Curve and Scatter Plot
    # We use matplotlib to visualize the TPR vs FAR.
    # =========================================================================
    
    # ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(false_alarm_rates, true_positive_rates, marker='o', linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='-') # Coin toss line
    plt.title('Receiver Operator Characteristic (ROC) Curve')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

    # 2. Plot the Scatter Plot
    print("Generating Scatter Plot...")
    
    # Add slight vertical "jitter" so the dots don't sit perfectly on top of each other
    intent_1_y = [1 + random.uniform(-0.05, 0.05) for _ in balanced_non_aggressive]
    intent_2_y = [2 + random.uniform(-0.05, 0.05) for _ in balanced_aggressive]
    
    plt.figure(figsize=(10, 3))
    
    # Plot the two classes using your exact balanced lists
    plt.scatter(balanced_non_aggressive, intent_1_y, color='blue', label='Non-Aggressive (1)', alpha=0.3, edgecolors='none', s=30)
    plt.scatter(balanced_aggressive, intent_2_y, color='red', label='Aggressive (2)', alpha=0.3, edgecolors='none', s=30)
    
    # Draw a vertical line for your dynamic best threshold
    plt.axvline(x=best_speed_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal Threshold (>= {best_speed_threshold})')
    
    # Format the chart with titles and labels
    plt.title('Scatter Plot of Driver Speed vs. Intent')
    plt.xlabel('Quantized Speed (mph)')
    plt.ylabel('Driver Intent (with slight vertical jitter)')
    plt.yticks([1, 2], ['1 (Non-Aggressive)', '2 (Aggressive)'])
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    
    # Save the image directly to your folder and then show it
    plt.savefig('scatter_plot_le_huy.png', dpi=300, bbox_inches='tight')
    print("Scatter plot successfully saved as 'scatter_plot_le_huy.png'!")
    plt.show()

    # =========================================================================
    # STEP 3B: Generating the Classifier Program
    # We dynamically create a new Python script that uses our discovered 
    # best_speed_threshold to classify new validation data.
    # =========================================================================
    
    classifier_code = f"""# =============================================================================
# Program Name: HW_03_Classifier_Le_Huy.py
# Purpose: This program was automatically generated by the Trainer program. 
#          It reads validation data and classifies drivers using the fixed 
#          threshold discovered during training.
# =============================================================================
import csv
import sys

def classify_data(input_filename):

    \"\"\"
    Reads validation data and classifies driver intent using a fixed speed threshold.

    This function opens a provided CSV file containing driving records, extracts 
    the 'Speed' attribute, and quantizes it to the nearest integer. It then applies 
    a hardcoded threshold (discovered during the training phase) to predict if the 
    driver is non-aggressive (1) or aggressive (2). The predictions are printed to 
    the console and saved as a new column in an output CSV file.

    Args:
        input_filename (str): The relative or absolute file path to the 
            validation CSV file. The CSV must contain a header row with a 
            'Speed' column.

    Returns:
        None: The function does not return a value. Instead, it writes the 
            results to 'HW_03_Le1_MyClassifications.csv' and prints 
            the classifications to standard output.

    Raises:
        FileNotFoundError: If the provided `input_filename` does not exist.
        ValueError: If the 'Speed' column contains data that cannot be 
            converted to a float.
    \"\"\"

    output_filename = 'HW_03_Le1_MyClassifications.csv'
    
    with open(input_filename, mode='r') as in_file, open(output_filename, mode='w', newline='') as out_file:
        csv_reader = csv.reader(in_file)
        csv_writer = csv.writer(out_file)
        
        header_row = next(csv_reader)
        # Add our guessed intent to the header
        header_row.append('Guessed_Intent')
        csv_writer.writerow(header_row)
        
        speed_index = header_row.index('Speed')
        
        for current_row in csv_reader:
            # Must quantize exactly the same way as the trainer!
            speed_val = int(float(current_row[speed_index])+0.5)
            
            # Apply the fixed threshold
            if speed_val >= {best_speed_threshold}:
                guessed_intent = 2 # aggressive
            else:
                guessed_intent = 1 # non-aggressive
                
            current_row.append(guessed_intent)
            csv_writer.writerow(current_row)
            
            # Print to standard output as required
            print(guessed_intent)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the validation CSV filename as a parameter.")
    else:
        classify_data(sys.argv[1])
"""

    classifier_filename = 'HW_03_Classifier_Le_Huy.py'
    with open(classifier_filename, mode='w') as classifier_file:
        classifier_file.write(classifier_code)
        
    print(f"Successfully generated the classifier: {classifier_filename}")

if __name__ == "__main__":
    main()