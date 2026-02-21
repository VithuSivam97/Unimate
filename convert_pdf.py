import pdfplumber
import pandas as pd
import os

pdf_path = "data/ugc_z_scores_2024_2025.pdf"
csv_path = "data/z_scores.csv"

def convert_pdf_to_csv():
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"Extracting tables from {pdf_path}...")
    
    all_rows = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                print(f"Processing page {i+1}...")
                tables = page.extract_tables()
                
                for table in tables:
                    for row in table:
                        # Clean row data
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        # Filter out empty rows or rows that look like headers/footers if needed
                        if any(cleaned_row):
                            all_rows.append(cleaned_row)
        
        if not all_rows:
            print("No tables found in the PDF.")
            return

        print(f"Found {len(all_rows)} rows. Saving to CSV...")
        
        # Create DataFrame
        # Assuming first row might be header, but for Z-scores usually it's complex. 
        # We'll just dump it all and let the user/LLM figure it out for now.
        df = pd.DataFrame(all_rows)
        
        df.to_csv(csv_path, index=False, header=False)
        print(f"Successfully saved to {csv_path}")
        
    except ImportError:
         print("Error: pdfplumber not installed. Please run: pip install pdfplumber")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_pdf_to_csv()
