import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM
from keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences


imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
dataTraining, dataTesting = imdb['train'], imdb['test']
xTraining = []
yTraining = []
xTesting = []
yTesting = []

class ValidationAccuracyStop(Callback):
    def __init__(self, accuracy=0.83):
        super(ValidationAccuracyStop, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= self.accuracy and logs.get('val_accuracy') >= self.accuracy:
            print("\nAccuracy of %0.2f%% reached, stopping training." % (self.accuracy * 100))
            self.model.stop_training = True
callbacks = ValidationAccuracyStop()

# DO NOT CHANGE THIS CODE
for s, l in dataTraining:
    xTraining.append(s.numpy().decode('utf8'))
    yTraining.append(l.numpy())

for s, l in dataTesting:
    xTesting.append(s.numpy().decode('utf8'))
    yTesting.append(l.numpy())

training_labels_final = np.array(yTraining)
testing_labels_final = np.array(yTesting)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

texts = xTraining

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(
    padded_sequences,
    training_labels_final,
    epochs=30,
    validation_split=0.2,
    verbose=1,
    callbacks=[callbacks])
