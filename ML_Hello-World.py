import numpy as np
import tensorflow as tf
from tensorflow import keras
X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                2.0, 3.0, 4.0, 5.0], dtype=float)
Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, ], dtype=float)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_dim=1)
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics = ['mse']
)
model.fit(X,Y, epochs=1500)
model.predict([-2.0, 10.0])

