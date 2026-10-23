from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data_files = list((ROOT / "data").rglob("*.xls")) + list((ROOT / "data").rglob("*.xlsx"))
if not data_files:
    raise FileNotFoundError("Put the Excel dataset inside the data folder.")

df = pd.read_excel(data_files[0], header=1)
target = "default payment next month"
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget counts:\n", df[target].value_counts())

(ROOT / "reports").mkdir(exist_ok=True)
ax = df[target].value_counts().sort_index().plot(kind="bar")
ax.set_title("Next-month payment default: class counts")
ax.set_xlabel("Target (0 = no default, 1 = default)")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "target_distribution.png", dpi=150)
plt.close()
print("\nSaved reports/target_distribution.png")
