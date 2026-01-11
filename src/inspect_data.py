import pandas as pd
from pathlib import Path

data_dir = Path("D:/kharagpur_hackathon/data")

# Load train.csv
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

print("="*60)
print("TRAIN.CSV STRUCTURE")
print("="*60)
train_df = pd.read_csv(train_path)
print(f"Shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print("\nFirst row:")
print(train_df.iloc[0].to_dict())

print("\n" + "="*60)
print("TEST.CSV STRUCTURE")
print("="*60)
test_df = pd.read_csv(test_path)
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print("\nFirst row:")
print(test_df.iloc[0].to_dict())

print("\n" + "="*60)
print("NOVELS IN FOLDER")
print("="*60)
novels_dir = data_dir / "novels"
for novel in novels_dir.iterdir():
    print(f"{novel.name}: {novel.stat().st_size / 1024 / 1024:.2f} MB")
    # Preview first 500 chars
    if novel.suffix == '.txt':
        content = novel.read_text(encoding='utf-8', errors='ignore')
        print(f"Preview: {content[:200]}...\n")
