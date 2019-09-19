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
        return [np.expand_dims(i, axis=0) for i in state]

    def make_model(self, state_size, action_size):
        field = Input(shape=state_size[0], name='field')
        current_puyo = Input(
            shape=state_size[1],
            name='current_puyo')
        next_puyo = Input(
            shape=state_size[2],
            name='next_puyo')
        next_next_puyo = Input(
            shape=state_size[3],
            name='next_next_puyo')

        x = Flatten()(field)
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
    d.run()
