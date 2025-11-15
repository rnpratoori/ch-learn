import numpy as np
import sys
from pathlib import Path

def read_npz_data(file_path):
    """
    Reads an .npz file and prints the keys and shapes of the arrays within it.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded {file_path}")
        print("Contents of the .npz file:")
        for key in data.keys():
            print(f"  Key: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")
        data.close()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_npz_data.py <path_to_npz_file>")
        sys.exit(1)

    npz_file_path = Path(sys.argv[1])
    read_npz_data(npz_file_path)
