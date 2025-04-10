def remove_backslashes(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line.replace('\\', ''))

# Specify your file paths
input_file = r'C:/Users/janak/AI/scienceProject2024/testData_max_values.csv'
output_file = r'C:/Users/janak/AI/scienceProject2024/DeepLearWing_v2.csv'

# Call the function
remove_backslashes(input_file, output_file)

print("Backslashes removed successfully!")
