import argparse
import json

def count_samples(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    total_samples = sum(shard['samples'] for shard in data['shards'])
    return total_samples

def main():
    parser = argparse.ArgumentParser(description="Count total samples from a JSON file containing shard information.")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()

    try:
        total_samples = count_samples(args.file_path)
        print(f"Total number of samples: {total_samples}")
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: '{args.file_path}' is not a valid JSON file.")
    except KeyError:
        print("Error: The JSON file does not have the expected structure.")

if __name__ == "__main__":
    main()