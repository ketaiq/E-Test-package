"""
This file generates a glossary of words from a Stackoverflow dataset.
"""

import os
import re
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords


def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

def count_words(folder_path):
    word_freq = Counter()
    # Exclude buzzwords such as articles, prepositions, and conjunctions
    nltk.download('stopwords')
    buzzwords = stopwords.words('english')
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print("Reading all data.")
            with open(file_path, 'rb') as file:
                content = file.read()
                # Decode file content using utf-8 encoding with error handling
                text = content.decode('utf-8', errors='replace')
                
                text = clean_text(text)
                # Split text into words
                words = text.split()
                
                # Count word frequencies
                word_freq.update(word for word in words if word not in buzzwords)
    return word_freq

def save_word_freq(word_freq, output_file):
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(sorted_word_freq, file, indent=4)

def main():
    # Data from: https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate?resource=download
    # 10%: https://www.kaggle.com/datasets/stackoverflow/stacksample/data
    folder_path = "../data"
    word_freq = count_words(folder_path)
    output_file = "./AutonomicTester/src/glossary/glossary_stackoverflow.json"
    save_word_freq(word_freq, output_file)
    print(f"Word frequencies saved to {output_file}")

if __name__ == "__main__":
    print("Generating glossary...")
    main()
