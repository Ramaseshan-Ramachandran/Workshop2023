from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(level=logging.INFO)

class NGramLM:
    def __init__(self, corpus_file, model_file, ngram_size):
        self.corpus_file = corpus_file
        self.ngram_size = ngram_size
        self.model_file = model_file
        self.model = None

    def build_model(self, embedding_size=100, window=5, min_count=1, workers=4):
        # Build the Word2Vec model
        self.model = Word2Vec(LineSentence(self.corpus_file), vector_size=embedding_size, window=window, min_count=min_count, workers=workers)
        self.save_model()

    def save_model(self):
        # Save the Word2Vec model
        if self.model:
            self.model.save(self.model_file)
            print("Model saved successfully.")
        else:
            print("No model available to save.")
    
    def load_model(self):
        self.model = Word2Vec.load(self.model_file)

    def predict_next_word(self, context):
        # Predict the next word based on the context
        self.load_model()
        if self.model:
            next_word = self.model.predict_output_word(context, topn=5)
            return next_word
        else:
            return []

# Example usage
if __name__ == '__main__':
    corpus_file = './corpus/budget_speech_en.txt'
    # corpus_file = './corpus/COVID19All.txt'
    ngram_size = 2
    model_file = './Models/AnnLM.bin'  # Path to save the trained model

    lm = NGramLM(corpus_file, model_file, ngram_size)
    lm.build_model()

    while True:
        prompt = tuple(input('Input context: ').lower().split())
        print(lm.predict_next_word(prompt))


# For fun
# generate_sentences using this model and find out 
# whether this gives any meaningful sentence