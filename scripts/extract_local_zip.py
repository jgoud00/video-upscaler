import zipfile
from pathlib import Path

def extract_div2k():
    zip_path = Path(r"C:\Users\Jaswant\Downloads\DIV2K_train_HR.zip")
    extract_to = Path(r"e:\machine learning\video-enhancer\data\train")
    
    print(f"Extracting {zip_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # zip_ref.extractall() will create the DIV2K_train_HR folder inside data/train
            zip_ref.extractall(extract_to)
        print("Extraction successful!")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    extract_div2k()
