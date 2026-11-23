from pathlib import Path
from datasets import load_dataset

output = Path("data/omnifall/annotations")
output.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "simplexsigil2/omnifall",
    "of-sta-cs",
)

print(ds)
print(ds["train"][0])
print(ds["train"].features)

for split_name, split_data in ds.items():
    file_path = output / f"{split_name}.csv"
    split_data.to_csv(str(file_path))
    print(f"Đã lưu: {file_path.resolve()}")