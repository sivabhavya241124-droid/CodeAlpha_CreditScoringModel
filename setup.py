from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# Project folders
for folder in ["data", "models", "src"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Download the dataset
url = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

zip_path = Path("data/credit_dataset.zip")

try:
    if not zip_path.exists():
        print("Downloading dataset...")
        urlretrieve(url, zip_path)

    print("Extracting dataset...")

    with ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.lower().endswith(".xls"):
                archive.extract(name, "data")
                print("Dataset extracted:", name)

    print("\nProject folders are ready!")
    print("Dataset download and extraction completed.")

except Exception as error:
    print("\nSetup failed:", error)
    print("Check your internet connection and try again.")