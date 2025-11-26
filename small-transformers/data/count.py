import os
from tqdm import tqdm

CORPUS_FILE = "data/corpus.txt"

def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: {CORPUS_FILE} not found.")
        return

    print(f"Counting words in {CORPUS_FILE}...")
    total_words = 0
    
    # Open file and count words line by line
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting"):
            words = line.split()
            total_words += len(words)

    print(f"Total words: {total_words:,}")

if __name__ == "__main__":
    main()
