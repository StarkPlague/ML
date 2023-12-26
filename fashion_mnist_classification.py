import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# End with 10 Neuron Dense, activated by softmax

# COMPILE MODEL HERE
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
# TRAIN YOUR MODEL HERE
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
