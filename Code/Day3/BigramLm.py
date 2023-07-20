import nltk
from nltk import bigrams
import dill
from tqdm import tqdm
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

import collections
from collections import Counter
from nltk import sent_tokenize, word_tokenize

corpus_dir = './TXT'  # Directory of corpus.
model_file = './Models/BigramLM.bin'


def sentence_tokenizer():
    sentences = []
    files = os.listdir(corpus_dir)
    for filename in tqdm(files, desc='Reading files', unit=' file'):
        with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
            text = file.read().lower()
            sentences = sent_tokenize(text)
            sentences.extend(sentences)
    return sentences

#compute the bigram model
def build_bigram_model():
    bigram_model = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0))
    
    sentences = sentence_tokenizer()
    for sentence in tqdm(sentences, desc='Sentences', unit=' sentence'):
        sentence = [word.lower() for word in word_tokenize(sentence)
                    if word.isalpha()]  # get alpha only
        #Collect all bigrams counts for (w1,w2)
        for w1, w2 in bigrams(sentence):
            bigram_model[w1][w2] += 1
        #compute the probability for the bigram starting with w1
        for w1 in bigram_model:
            #total count of bigrams starting with w1
            total_count = float(sum(bigram_model[w1].values()))
            #distribute the probability mass for all bigrams starting with w1
            for w2 in bigram_model[w1]:
                bigram_model[w1][w2] /= total_count

    print('Saving the model....')
    save_model(bigram_model)

    return bigram_model

def save_model(model):
    with open(model_file, 'wb') as file:
        dill.dump(model, file)
        file.close()

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = dill.load(file)
        file.close()
        return model  

def predict_next_word(first_word):
    model = load_model(model_file)

    #get the next for the bigram starting with 'word'
    second_word = model[first_word]
    #get the top 10 words whose first word is 'first_word'
    top10words = Counter(second_word).most_common(10)


    predicted_words = list(zip(*top10words))[0]
    probability_score = list(zip(*top10words))[1]
    x_pos = np.arange(len(predicted_words))

    plt.bar(x_pos, probability_score,align='center')
    plt.xticks(x_pos, predicted_words)
    plt.ylabel('Probability Score')
    plt.xlabel('Predicted Words')
    plt.title('Predicted words for ' + first_word)
    plt.show()

# build_bigram_model()

while (1):
    word = input('Input first word: ')
    predict_next_word(word) #development, water
