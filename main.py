from collections import deque

import keras
import numpy as np
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

from dqn import DQN
from puyopuyo import PuyoPuyo


class PuyoPuyoDQN(DQN):
    def state2input(self, state):
        if isinstance(state, tuple):
            field, puyo1, puyo2, puyo3 = state
            return [
                np.expand_dims(np.expand_dims(field, axis=0), axis=3),
                np.expand_dims(puyo1, axis=0),
                np.expand_dims(puyo2, axis=0),
                np.expand_dims(puyo3, axis=0)
            ]
        else:
            return [
                np.expand_dims(np.array([field for (field, _, _, _) in state], dtype=np.uint8), axis=3),
                np.array([puyo for (_, puyo, _, _) in state], dtype=np.uint8),
                np.array([puyo for (_, _, puyo, _) in state], dtype=np.uint8),
                np.array([puyo for (_, _, _, puyo) in state], dtype=np.uint8)
            ]

    def make_model(self, state_size, action_size):
        field = Input(shape=(*state_size[0], 1), name='field')
        current_puyo = Input(
            shape=state_size[1],
            name='current_puyo')
        next_puyo = Input(
            shape=state_size[2],
            name='next_puyo')
        next_next_puyo = Input(
            shape=state_size[3],
            name='next_next_puyo')
        x = Conv2D(32, 2)(field)
        x = Conv2D(32, 2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        x = keras.layers.concatenate(
            [x, current_puyo, next_puyo, next_next_puyo])
        output = Dense(action_size, activation='sigmoid', name='output')(x)
        model = Model(inputs=[field, current_puyo, next_puyo, next_next_puyo],
                      output=output)
        model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
        return model


if __name__ == "__main__":
    d = PuyoPuyoDQN(PuyoPuyo())
    d.run(num_episodes=500, max_steps=200, batch_size=8)
