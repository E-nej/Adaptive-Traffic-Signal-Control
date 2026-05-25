# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:31:15 2019

@author: xz01m2
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class Model:
    def __init__(self, num_states, num_actions):
        model = Sequential([
            Input(shape=(num_states,)),
            Dense(400),
            Dense(400, activation='relu'),
            Dense(num_actions, activation='linear'),
        ])
        model.compile(loss='huber', optimizer=Adam(learning_rate=2.5e-4, clipnorm=1.0))
        self.model = model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def predict(self, state):
        # Direct model call avoids Keras predict pipeline overhead and tf.function retracing
        return self.model(state, training=False).numpy()

    def save(self, filename):
        self.model.save(filename)
