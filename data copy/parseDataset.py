import csv
import random

def remove_random_rows(input_file, output_file, rows_to_remove):
    """
    Removes a set number of random rows (excluding the header) from a CSV file and writes the result to a new file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        rows_to_remove (int): The number of rows to randomly remove (excluding the header).
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)  # Read all rows into a list

    # Randomly select rows to remove
    rows_to_keep = random.sample(range(len(rows)), len(rows) - rows_to_remove)

    # Write the data back to the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        for row_index in rows_to_keep:
            writer.writerow(rows[row_index])

# Example usage
input_csv = "C:/Users/janak/AI/scienceProject2024/data/DeepLearWing_v2.csv"   # Replace with your input file
output_csv = "C:/Users/janak/AI/scienceProject2024/data/simpleData.csv" # Replace with your output file

# Randomly remove 750,000 rows (keeping the header intact)
remove_random_rows(input_csv, output_csv, rows_to_remove=700000)

print(f"Random 700,000 rows removed and saved to {output_csv}")
