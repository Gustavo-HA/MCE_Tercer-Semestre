import pandas as pd
import argparse
import logging
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
logger = logging.Logger(__name__)

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    return data

def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text.encode('latin-1').decode('utf-8', errors='ignore')
    return s

def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    """ No utilizaremos las columnas 'Town', 'Region' y 'Type' """
    data = data.drop(columns=['Town', 'Region', 'Type'], errors='ignore')
    data['Review'] = data['Review'].astype(str).apply(fix_mojibake)
    data['Polarity'] = data['Polarity'].astype(int) - 1
    data = data.rename(columns={'Review': 'text', 'Polarity': 'label'})
    data['uuid'] = data.index
    return data

def main():
    parser = argparse.ArgumentParser(description='Prepare data for BERT classification')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the processed JSON file')
    args = parser.parse_args()

    logger.info(f"Loading data from {args.file_path}")
    data = load_data(args.file_path)
    
    logger.info("Preprocessing text data")
    data = preprocess_text(data)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['label'])

    logger.info(f"Saving processed data to {args.output_path}")

    data_dict = {
        'train': train_data.to_dict(orient='records'),
        'validation': val_data.to_dict(orient='records'),
        'test': test_data.to_dict(orient='records')
    }

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

        
    logger.info(f"Data saved to {args.output_path}")
    
    
if __name__ == "__main__":
    main()