import csv
import random
import os

# Path to the dataset
# Using the absolute path found in the system
DATA_PATH = r"C:\Users\Jad Zoghaib\OneDrive\Desktop\CC_Breast_Cancer\Breast-cancer-risk-prediction\data\clean-data.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(path):
    malignant = []
    benign = []
    
    if not os.path.exists(path):
        print(f"Error: Could not find data file at {path}")
        return [], []

    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            for row in reader:
                # row structure: [id, diagnosis, feature1, feature2, ... feature30]
                if len(row) < 32: # Ensure we have enough columns
                    continue
                    
                # Use the ID from the file (row[0]) or the line number?
                # The file has an explicit ID column at index 0.
                case_id = row[0]
                diagnosis = row[1]
                features = row[2:] 
                
                # Convert features to floats
                try:
                    features = [float(x) for x in features]
                except ValueError:
                    continue 
                
                case = {'id': case_id, 'features': features, 'diagnosis': diagnosis}
                
                if diagnosis == 'M':
                    malignant.append(case)
                elif diagnosis == 'B':
                    benign.append(case)
                    
        return malignant, benign
    except Exception as e:
        print(f"Error reading data: {e}")
        return [], []

def save_case(case, custom_id=None):
    # Filename format: sample_patient_{ID}.csv
    # Use custom_id if provided, otherwise use case's original id
    file_id = custom_id if custom_id else case['id']
    filename = f"sample_patient_{file_id}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            # Write features as a single row, comma separated
            f.write(",".join(map(str, case['features'])))
        print(f"Generated {case['diagnosis']} case: {filename}")
    except Exception as e:
        print(f"Error writing file {filename}: {e}")

def main():
    print(f"Reading data from: {DATA_PATH}")
    malignant, benign = load_data(DATA_PATH)
    
    if not malignant or not benign:
        print("Failed to load data or data is empty.")
        return

    print(f"Found {len(malignant)} Malignant and {len(benign)} Benign cases.")
    
    while True:
        print("\n--- New Case Generation ---")
        custom_id = input("Enter Case ID (or press Enter to exit): ").strip()
        if not custom_id:
            break
            
        print(f"Selected ID: {custom_id}")
        print("Choose data source:")
        print("1. Random Malignant Case")
        print("2. Random Benign Case")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            save_case(random.choice(malignant), custom_id)
        elif choice == '2':
            save_case(random.choice(benign), custom_id)
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
