import os
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import json

nltk.download('punkt')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
DATA_PATH = os.path.join(ROOT_DIR, "dataset", "AllCombined.txt")
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_data")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_wikipedia_subset(file_path, max_chars=5_000_000):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_chars)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_chunking(text, max_words=150):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
        current_chunk.append(sent)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def create_documents(chunks):
    documents = [
        {"id": i, "text": chunk, "source": "SimpleEnglish Wikipedia"}
        for i, chunk in enumerate(chunks)
    ]
    return documents

def save_processed_data(cleaned_text, chunks, documents):
    with open(os.path.join(PROCESSED_DIR, "cleaned_text.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    np.save(os.path.join(PROCESSED_DIR, "chunks.npy"), np.array(chunks))
    
    # Save documents metadata as JSON
    with open(os.path.join(PROCESSED_DIR, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

def main():
    raw_text = load_wikipedia_subset(DATA_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = sentence_chunking(cleaned_text)
    documents = create_documents(chunks)
    save_processed_data(cleaned_text, chunks, documents)
    print(f"Processed {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
