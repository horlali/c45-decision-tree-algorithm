from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

household_data = DATA_DIR / "raw" / "household_data.csv"
household_test_data = DATA_DIR / "raw" / "household_data_test.csv"

transformed_data = DATA_DIR / "transformed" / "transformed_data.csv"
transformed_test_data = DATA_DIR / "transformed" / "transformed_data_test.csv"
