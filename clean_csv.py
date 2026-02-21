import csv
import re

input_path = "data/z_scores.csv"
output_path = "data/cleaned_z_scores.csv"

def is_valid_district_row(row):
    # Valid rows generally start with a district name (UPPERCASE) and have > 5 columns of numbers
    if not row: return False
    first_cell = row[0].strip()
    
    # Check if first cell is a known district or looks like one (Upppercase, >3 chars)
    if not (first_cell.isupper() and len(first_cell) > 3):
        return False
        
    # Check if subsequent cells contain numbers
    number_count = 0
    for cell in row[1:]:
        if re.search(r'\d+\.\d+', str(cell)):
            number_count += 1
            
    return number_count > 3  # Arbitrary threshold

try:
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Add a likely header row manually since the original is probably garbage
        # Based on typical Z-score tables: District, Bio, Phy, Che, etc.
        # Since we don't know the exact order, we'll label them generically for the LLM to infer from context or just leave them raw
        # Better: Let's just keep the valid data rows. The LLM can fuzzy match "Engineering" columns if they have high values vs Arts.
        
        valid_rows = []
        for row in reader:
            if is_valid_district_row(row):
                valid_rows.append(row)
        
        if valid_rows:
            print(f"Found {len(valid_rows)} valid data rows.")
            writer.writerows(valid_rows)
        else:
            print("No valid rows found using heuristics.")
            
except Exception as e:
    print(f"Error cleaning CSV: {e}")
