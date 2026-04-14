import pickle

def inspect_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded object type: {type(data)}\n")

    # If it's a pandas DataFrame
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            print("Columns (keys):")
            print(list(data.columns), "\n")

            print("Head (first 5 rows):")
            print(data.head(), "\n")

            print("Tail (last 5 rows):")
            print(data.tail())
            return
    except ImportError:
        pass

    # If it's a dictionary
    if isinstance(data, dict):
        keys = list(data.keys())
        print("Keys:")
        print(keys, "\n")

        print("First 5 key-value pairs:")
        for k in keys[:5]:
            print(f"{k}: {data[k]}")
        print("\nLast 5 key-value pairs:")
        for k in keys[-5:]:
            print(f"{k}: {data[k]}")
        return

    # If it's a list or tuple
    if isinstance(data, (list, tuple)):
        print(f"Total length: {len(data)}\n")

        print("First 5 entries:")
        for item in data[:5]:
            print(item)

        print("\nLast 5 entries:")
        for item in data[-5:]:
            print(item)
        return

    # Fallback for other object types
    print("Object does not support head/tail directly.")
    print("Full object representation:")
    print(data)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <file.pkl>")
    else:
        inspect_pkl(sys.argv[1])
