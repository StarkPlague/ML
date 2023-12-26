from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM
from keras.callbacks import Callback
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ValidationAccuracyStop(Callback):
    def __init__(self, accuracy=0.92):
        super(ValidationAccuracyStop, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('acc') >= self.accuracy and logs.get('val_acc') >= self.accuracy:
            print("\nAccuracy of %0.2f%% reached, stopping training." % (self.accuracy * 100))
            self.model.stop_training = True
callbacks = ValidationAccuracyStop()
bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

# DO NOT CHANGE THIS CODE
# Make sure you used all of these parameters or you can not pass this test
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_portion = .8

# YOUR CODE HERE
# Using "shuffle=False"
sentences = bbc['text'].to_list()
labels = bbc['category'].to_list()
# Using "shuffle=False"
train_size = int(len(sentences) * training_portion)

train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(
    sentences,
    labels,
    train_size=train_size,
    shuffle=False
)

bbc['category'] = bbc['category'].replace({'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4})

# Fit your tokenizer with training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # YOUR CODE HERE
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length,
                                    truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])
model.fit(train_padded,
            training_label_seq,
            epochs=30,
            batch_size=20,
            validation_data=(validation_padded, validation_label_seq),
            callbacks=[callbacks])
