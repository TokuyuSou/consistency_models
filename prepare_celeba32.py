# prepare_celeba32_from_csv.py
import pandas as pd
from PIL import Image
from pathlib import Path

# Input paths
root = Path(
    "/workspace/celeba"
)  # replace with the path containing img_align_celeba and CSVs
img_dir = root / "img_align_celeba/img_align_celeba"
csv_file = root / "list_eval_partition.csv"

# Output root
out_root = Path("/workspace/consistency_models/datasets/celeba32")
out_root.mkdir(parents=True, exist_ok=True)

# Load split info (0=train, 1=valid, 2=test)
df = pd.read_csv(csv_file)

for idx, row in df.iterrows():
    filename, split_id = row[0], row[1]
    split = {0: "train", 1: "valid", 2: "test"}[split_id]

    # Open and resize to 32x32
    img = Image.open(img_dir / filename).convert("RGB")
    img = img.resize((32, 32), Image.BICUBIC)

    # Save under ImageFolder structure: split/all/
    out_dir = out_root / split / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    img.save(out_path)

print("Done! CelebA resized to 32x32 and split into train/valid/test.")
