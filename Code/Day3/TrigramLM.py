import nltk
from nltk import bigrams, trigrams
import dill
from tqdm import tqdm
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

import collections
from collections import Counter
from nltk import sent_tokenize, word_tokenize

corpus_dir = './TXT'  # Directory of corpus.
model_file = './Models/TrigramLM.bin'

#read every file, split into sentences and concate them
def sentence_tokenizer():
    sentences = []
    files = os.listdir(corpus_dir)
    for filename in tqdm(files, desc='Reading files', unit=' file'):
        with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
            text = file.read().lower()
            sentences = sent_tokenize(text)
            sentences.extend(sentences)
    return sentences


def build_trigram_model():
    model = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    sentences = sentence_tokenizer()
    #For every sentence, find the trigram
    for sentence in tqdm(sentences, desc='Sentences', unit=' sentence'):
        
        sentence = [word for word in word_tokenize(sentence)
                    if word.isalpha()]  # get alpha only
        
        #increment the count using context and the third word
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

        #count the total number of context (w1 and w2)
        for w1_w2 in model:
            total_count = float(sum(model[w1_w2].values()))
            #compute the probability (MLE)
            for w3 in model[w1_w2]:
                model[w1_w2][w3] /= total_count

    print('\nWriting model file')
    save_model(model)

def save_model(model):
    with open(model_file, 'wb') as file:
        dill.dump(model, file)
    file.close()

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = dill.load(file)  
        file.close()
        return model

def predict_next_word(model, w1,w2):
    #find the sent of next words given the context P(w3|w1,w2)
    next_word = model[(w1,w2)]

    predicted_words = collections.Counter(next_word).most_common(10)

    top10Predicted_words = list(zip(*predicted_words))[0]

    print(top10Predicted_words)


    # plt.bar(x_pos, probability_score,align='center')
    # plt.xticks(x_pos, top10Predicted_words)
    # plt.title('Predicted words for  '+ w1 + ' ' + w2)
    # plt.ylabel('Probability Score')
    # plt.xlabel('Predicted Words')
    # plt.show()

# build_trigram_model()
model = load_model(model_file)
query = ''
while (1):
    query = input('Next words for (w1,w2) - ')
    w1,w2 = query.split()
    predict_next_word(model,w1,w2) 
    #(from, the), (rural, development), (income, tax)

# predict_next_word(model,'none', 'none')
