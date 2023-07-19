import nltk
from nltk.util import ngrams

def extract_bigrams(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Generate the bigrams
    bigrams = list(ngrams(tokens, 2))
    # What happens when use
    # bigrams = ngrams(tokens,2,
    #                 pad_left=True, pad_right=True,
    #                 left_pad_symbol='<s>', 
    #                 right_pad_symbol='</s>')
    
    return bigrams

# Sample text
sample_text = "Natural language processing (NLP) is a subfield of artificial intelligence. \
                It focuses on the interaction between computers and human language. \
                NLP techniques are used in various applications such as machine translation and sentiment analysis."

# Extract bigrams from the sample text
result = extract_bigrams(sample_text)

# Print the resulting bigrams
for bigram in result:
    print(bigram)
