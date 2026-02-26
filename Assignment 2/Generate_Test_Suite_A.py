import csv

# =============================================================================
# Script Name: Generate_Test_Suite_A.py
# Purpose: Generates a small, balanced test suite of 8 records for a threshold 
#          classifier. This allows us to test our mentor program with known 
#          data where the "speed" attribute perfectly separates the classes.
# =============================================================================

def generate_test_suite():
    """
    Creates a CSV file named 'Test_suite_A_speed.csv' with 8 records.
    The first column represents the 'Speed' attribute, and the last column 
    represents the target attribute 'Intent' (1 = non-aggressive, 2 = aggressive).
    """
    
    # Define the filename for the test suite
    output_filename = 'Test_suite_A_speed.csv'
    
    # Define the headers for the CSV file. 
    # The last attribute is the target attribute (Intent).
    headers = ['Speed', 'Intent']
    
    # Create the perfectly separable dataset:
    # - The first 4 records are non-aggressive (Intent = 1) with lower speeds.
    # - The last 4 records are aggressive (Intent = 2) with higher speeds.
    # A threshold of 65 will cleanly split this data.
    test_data = [
        [40, 1],
        [45, 1],
        [50, 1],
        [55, 1],
        [75, 2],
        [80, 2],
        [85, 2],
        [90, 2]
    ]
    
    # Open the file in write mode and output the data as comma-separated values
    with open(output_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow(headers)
        
        # Write the 8 records to the file
        csv_writer.writerows(test_data)
        
    print(f"Successfully generated '{output_filename}' with {len(test_data)} records.")

# Execute the function if this script is run directly
if __name__ == "__main__":
    generate_test_suite()