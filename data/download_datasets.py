# data/download_datasets.py
import os
import subprocess
import logging
from pathlib import Path
import gdown  # Requires pip install gdown

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_esc():
    """Download ESC dataset (40,932 Ethereum smart contracts) from Google Drive."""
    esc_url = 'https://drive.google.com/uc?id=1yFJSCiUuoiSx4uWYNcCESUvsEs5DOGM9'
    esc_output = RAW_DIR / 'esc_dataset.zip'
    if not esc_output.exists():
        logging.info("Downloading ESC dataset...")
        gdown.download(esc_url, str(esc_output), quiet=False)
        # Unzip (assuming it's zipped; adjust if not)
        subprocess.run(['unzip', '-o', str(esc_output), '-d', str(RAW_DIR / 'esc')])
        logging.info("ESC dataset downloaded and extracted to data/raw/esc")
    else:
        logging.info("ESC dataset already exists.")

def download_vsc():
    """Download VSC dataset (4,170 VNT Chain smart contracts) from Google Drive."""
    vsc_url = 'https://drive.google.com/uc?id=1FTb__ERCOGNGM9dTeHLwAxBLw7X5Td4v'
    vsc_output = RAW_DIR / 'vsc_dataset.zip'
    if not vsc_output.exists():
        logging.info("Downloading VSC dataset...")
        gdown.download(vsc_url, str(vsc_output), quiet=False)
        # Unzip
        subprocess.run(['unzip', '-o', str(vsc_output), '-d', str(RAW_DIR / 'vsc')])
        logging.info("VSC dataset downloaded and extracted to data/raw/vsc")
    else:
        logging.info("VSC dataset already exists.")

def download_solidifi():
    """Clone SolidiFI benchmark dataset from GitHub."""
    solidifi_dir = RAW_DIR / 'solidifi'
    if not solidifi_dir.exists():
        logging.info("Cloning SolidiFI benchmark...")
        subprocess.run(['git', 'clone', 'https://github.com/DependableSystemsLab/SolidiFI-benchmark.git', str(solidifi_dir)])
        logging.info("SolidiFI dataset cloned to data/raw/solidifi")
    else:
        logging.info("SolidiFI dataset already exists.")

def main():
    try:
        download_esc()
        download_vsc()
        download_solidifi()
        logging.info("All datasets downloaded. Next steps: Preprocess using src/hogat/tools/normalization.py and graph_construction.py to generate data/processed/ graphs.")
    except Exception as e:
        logging.error(f"Error during download: {e}")

if __name__ == "__main__":
    main()