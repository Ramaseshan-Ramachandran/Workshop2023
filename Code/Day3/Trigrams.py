import nltk
from nltk.util import ngrams

def extract_trigrams(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Generate the trigrams
    trigrams = ngrams(tokens,3,
                    pad_left=True, pad_right=True,
                    left_pad_symbol='<s>', 
                    right_pad_symbol='</s>')    
    return trigrams

# Sample text
sample_text = "Natural language processing (NLP) is a subfield of artificial intelligence.\
                It focuses on the interaction between computers and human language. \
                NLP techniques are used in various applications such as machine translation and sentiment analysis."

# Extract trigrams from the sample text
trigrams = extract_trigrams(sample_text)

# Print the resulting trigrams
for trigram in trigrams:
    print(trigram)

print(len(list(zip(trigrams))))

#Exercise
#Use sent_tokenize to split the text into sentences
#Then use the word_tokenize to get the trigrams
#Why is this step required?
