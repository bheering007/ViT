import os, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("TestData_77HZ_12_14")
DST = Path("splits")

classes = [d.name for d in SRC.iterdir() if d.is_dir()]
files, labels = [], []

for c in classes:
    for f in (SRC / c).glob("*"):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            files.append(str(f))
            labels.append(c)

train_f, temp_f, train_y, temp_y = train_test_split(
    files, labels, test_size=0.3, stratify=labels, random_state=42
)
val_f, test_f, val_y, test_y = train_test_split(
    temp_f, temp_y, test_size=0.5, stratify=temp_y, random_state=42
)

def copy_split(fs, ys, split):
    for f, y in zip(fs, ys):
        out = DST / split / y
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / Path(f).name)

if DST.exists():
    shutil.rmtree(DST)

copy_split(train_f, train_y, "train")
copy_split(val_f, val_y, "val")
copy_split(test_f, test_y, "test")

print("Splits created.")
