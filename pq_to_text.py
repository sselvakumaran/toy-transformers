import pandas as pd
import os
import re # <-- NEW: Import for regular expression cleaning

# --- Configuration ---

# 1. List of your input Parquet files (assuming structure like data/raw/gsm8k/train.parquet)
base_path = "data/raw/gsm8k"
FILENAMES = [
    "train.parquet",
    "test.parquet",
]
# Full paths to the input files
INPUT_FILES = [os.path.join(base_path, f) for f in FILENAMES]

# 2. Name of the output text file
OUTPUT_FILENAME = "data/gsm8k/full.txt"

# --- Cleaning Helper ---

def clean_unusual_terminators(text):
    """
    Removes Line Separator (U+2028) and Paragraph Separator (U+2029) characters 
    from the text, replacing them with a space.
    """
    # Use regex to replace the specific unusual terminators with a standard space
    return re.sub(r'[\u2028\u2029]', ' ', text)

# --- Script Start ---

def join_parquet_files_to_text(input_files, output_filename):
    """
    Reads 'question' and 'answer' columns from multiple Parquet files, 
    formats them as Q\nA, and joins them into a single text file.
    """
    print(f"Starting consolidation of {len(input_files)} files...")
    
    # Check if the output directory exists, and create it if necessary
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Check if any input files are missing
    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: Input file not found: {f}")
            print("Please ensure your Parquet files exist at the specified paths.")
            return

    all_text_chunks = []
    
    for i, file_path in enumerate(input_files):
        print(f"Processing file {i + 1}/{len(input_files)}: {file_path}")
        
        try:
            # Read the Parquet file
            df = pd.read_parquet(file_path)
            
            # Define the required columns
            required_columns = ['question', 'answer']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: One or more required columns ('question', 'answer') not found in {file_path}.")
                return

            formatted_pairs = []
            
            # Iterate over the DataFrame to combine question and answer rows
            for _, row in df.iterrows():
                question = str(row['question']).strip()
                answer = str(row['answer']).strip()
                
                # NEW STEP: Apply cleaning to remove unusual line terminators
                question = clean_unusual_terminators(question)
                answer = clean_unusual_terminators(answer)
                
                # Format as 'question\nanswer'
                formatted_pairs.append(f"<START>{question}\n{answer}<END>")
            
            # Join all formatted pairs within this file with a blank line separator
            text_chunk = '\n\n'.join(formatted_pairs)
            all_text_chunks.append(text_chunk)
            
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
            return

    # Join all file chunks into the final content
    final_content = '\n\n'.join(all_text_chunks)
    
    # Write the combined content to the output file
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(final_content)

    print("-" * 40)
    print(f"✅ Success! All data written to '{output_filename}'")
    # Check file existence before trying to get its size
    if os.path.exists(output_filename):
        print(f"Total size (approx): {os.path.getsize(output_filename) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    join_parquet_files_to_text(INPUT_FILES, OUTPUT_FILENAME)