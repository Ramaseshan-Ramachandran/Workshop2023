import os
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import save_model, load_model
import re

# Step 1: Prepare the Dataset
corpus_folder = './/TXT'

def read_corpus(corpus_folder='./TXT'):
    corpus_text = ''
    for filename in os.listdir(corpus_folder):
        file_path = os.path.join(corpus_folder, filename)
        with open(file_path, "r") as f:
            corpus_text += f.read().lower()
    return corpus_text

# Step 2: Text Preprocessing

def process_text():
    corpus_text = read_corpus()
    filters = '!"#$%&()*+/:;<=>?@[\\]^_`{|}~'
    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts([corpus_text])

    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences([corpus_text])[0]
    return vocab_size, sequences, tokenizer

# Step 3: Prepare Training Data


def build_model():
    vocab_size, sequences,tokenizer = process_text()
    sequence_length = 10
    sequences = np.array(sequences)
    X = []
    y = []

    for i in range(sequence_length, len(sequences)):
        sequence = sequences[i-sequence_length:i]
        target = sequences[i]
        X.append(sequence)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Pad sequences if needed
    X = pad_sequences(X, maxlen=sequence_length)

    # Step 4: Build the Language Model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=sequence_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return X,y, model

def train():

    X,y,model = build_model()
    #we can add {epoch:02d}-{loss:.4f} to the filename to create epoch-wise chp files
    check_point_file = './Models/model_checkpoint.{epoch:02d}-{loss:.4f}.h5'
    if os.path.isfile('./Models/model_checkpoint.h5'):
      model.load_weights('./Models/model_checkpoint.h5')
    # Step 5: Train the Language Model

    checkpoint_callback = ModelCheckpoint(check_point_file,
                                        save_weights_only=False,
                                        save_best_only=True,
                                        monitor='val_loss')

    model.summary()

    # checkpoint_callback = ModelCheckpoint('./Models/model_checkpoint.h5',
    #                                       save_weights_only=False, save_best_only=True)
    # model.fit(X, y, batch_size=128, epochs=2, callbacks=[checkpoint_callback])
    model.fit(X, y, batch_size=128, epochs=20,
            validation_data=(X, y),
            callbacks=[checkpoint_callback])
    model.save('./Models/RNN_LM.hdf5')

    # model.save('/content/drive/MyDrive/Datasets/Budget_speech/Models/Keras_RNN.LM')

def generate_sentences(seed_text, num_sentences, sequence_length):
    _,_,tokenizer = process_text()
    _,_,model = build_model()
    # check_point_file = '/content/drive/MyDrive/Models/model_checkpoint.h5'
    if os.path.isfile('./Models/model_checkpoint.h5'):
      model.load_weights('./Models/model_checkpoint.h5')


    generated_text = seed_text + ': '
    for _ in range(num_sentences):
        for _ in range(sequence_length):
            input_sequence = tokenizer.texts_to_sequences([seed_text])[0]
            input_sequence = np.array(input_sequence)
            input_sequence = pad_sequences([input_sequence], maxlen=sequence_length)

            predicted_index = np.argmax(model.predict(input_sequence))
            predicted_word = tokenizer.index_word[predicted_index]

            seed_text += " " + predicted_word

            if predicted_word == '.':
                break

        generated_text += seed_text
        seed_text = predicted_word

    return generated_text


train()

seed_text = "budget speech"
num_sentences = 5
generated_text_length = 100
sequence_length = 10

# model = load_model('/content/drive/MyDrive/Datasets/Budget_speech/Models/Keras_RNN.LM')
generated_text = generate_sentences(seed_text, num_sentences, sequence_length)
print(generated_text)