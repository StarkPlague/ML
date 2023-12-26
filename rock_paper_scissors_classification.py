import urllib.request
import zipfile
import tensorflow as tf
import os
from keras.callbacks import Callback
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "data/rps/"
training_datagen = ImageDataGenerator(
    rescale=1. / 255)

# YOUR IMAGE SIZE SHOULD BE 150x150
# Make sure you used "categorical"
train_generator=training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=5,
    verbose=1,
)
