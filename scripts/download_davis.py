import subprocess
import zipfile
import os
from pathlib import Path

def download_and_extract(url, target_dir):
    zip_path = Path("davis.zip")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DAVIS dataset from {url}...")
    try:
        # Download the file using native curl (bypasses some python DNS issues on Windows)
        subprocess.run(["curl.exe", "-L", "-o", str(zip_path), url], check=True)
        print("Download complete. Extracting...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We only care about the JPEGImages folder inside the zip
            for file_info in zip_ref.infolist():
                if file_info.filename.startswith("DAVIS/JPEGImages/480p/"):
                    zip_ref.extract(file_info, target_dir)
                    
        print("Extraction complete!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up zip file.")

if __name__ == "__main__":
    davis_url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    target_directory = "data/train/davis"
    
    print("Starting DAVIS dataset prep...")
    download_and_extract(davis_url, target_directory)
    print(f"Done. Frames are ready in {target_directory}/DAVIS/JPEGImages/480p/")
