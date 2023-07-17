import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder

# Set the path to the folder containing the text files
folder_path = "./TXT"
# Load English stop words
stop_words = set(stopwords.words("english"))
# Process each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # Read and preprocess the text file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = word_tokenize(text)

        # Filter and keep only English words
        words = [w for w in tokens if w.isalpha() and w not in stop_words]

# Initialize the bigram collocation finder with the filtered words
bigram_finder = BigramCollocationFinder.from_words(words)

# Calculate the co-occurrence matrix - ngram frequency distribution
cooccur_matrix = bigram_finder.ngram_fd
print(cooccur_matrix.most_common(10))
#Exercise - print the concurrent matrix - pretty print only bigrams where the freq count > 10
#in this format
#       w1  w2  w3....
#w1     0   2   1
#w2     2   0   6  
#w3     1   6   0
#...    ... ..  ..  ...

# extend this to construct a trigram cooccurence table