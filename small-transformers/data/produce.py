import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "ThomasTheMaker/arc-fineweb"
OUTPUT_FILE = "data/corpus.txt"

def main():
    # Load the dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Open the output file
    print(f"Writing to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Iterate over the dataset and write the 'text' column
        # Using tqdm for a progress bar (though total might be unknown for streaming)
        for item in tqdm(dataset, desc="Processing"):
            text = item.get("text", "")
            if text:
                f.write(text + "\n")

    print("Done!")

if __name__ == "__main__":
    main()