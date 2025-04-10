import csv

# Define file paths
input_file = "C:/Users/janak/AI/scienceProject2024/data/simpleData.csv"
output_file = "C:/Users/janak/AI/scienceProject2024/CTGAN/coordinatesForCTGAN.csv"  # Output file for x and y coordinates

# Open the input file and process it
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL, escapechar='\\', delimiter=',')

    header = next(reader)  # Read header
    writer.writerow(["x_coords", "y_coords"])  # Write a new header for the output

    for row in reader:
        try:
            # Identify the start and end indices for x and y coordinates
            x_start_index = row.index("1.0")
            x_end_index = row.index("1.0", x_start_index + 1)
            y_start_index = x_end_index + 1
            y_end_index = len(row) - 6

            # Extract coordinates
            x_coords = row[x_start_index:x_end_index + 1]
            y_coords = row[y_start_index:y_end_index]

            # Write the entire x_coords and y_coords in separate columns for each airfoil
            writer.writerow([",".join(x_coords), ",".join(y_coords)])

        except (ValueError, IndexError) as e:
            # Skip rows that cannot be processed
            print(f"Skipping row due to error: {e}")

# Now remove quotation marks from the CSV file
with open(output_file, "r") as infile:
    content = infile.read()

# Replace all quotes (") in the file content
content_no_quotes = content.replace('"', '')

# Write the modified content back to the file
with open(output_file, "w") as outfile:
    outfile.write(content_no_quotes)

print(f"Processed CSV without quotation marks has been saved to {output_file}")
