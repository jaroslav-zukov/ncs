from pathlib import Path

package_dir = Path(__file__).resolve().parent

PROJECT_ROOT = package_dir.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"