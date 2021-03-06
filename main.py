import keras
import numpy as np
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, MaxPooling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

from dqn import DQN
from puyopuyo import PuyoPuyo


class PuyoPuyoDQN(DQN):
    def state2input(self, state):
        if isinstance(state, tuple):
            field, puyo1, puyo2, puyo3 = state
            field_input = [field == i + 1 for i in range(4)]
            return [
                np.expand_dims(np.stack(field_input, 2), axis=0),
                np.expand_dims(puyo1, axis=0),
                np.expand_dims(puyo2, axis=0),
                np.expand_dims(puyo3, axis=0)
            ]
        else:
            fields = np.array(
                [field for (field, _, _, _) in state], dtype=np.uint8)
            return [
                np.array([np.stack([field == i + 1 for i in range(4)], 2) for field in fields]),
                np.array([puyo for (_, puyo, _, _) in state], dtype=np.uint8),
                np.array([puyo for (_, _, puyo, _) in state], dtype=np.uint8),
                np.array([puyo for (_, _, _, puyo) in state], dtype=np.uint8)
            ]

    def make_model(self, state_size, action_size):
        field = Input(shape=(*state_size[0], 4), name='field')
        current_puyo = Input(
            shape=state_size[1],
            name='current_puyo')
        next_puyo = Input(
            shape=state_size[2],
            name='next_puyo')
        next_next_puyo = Input(
            shape=state_size[3],
            name='next_next_puyo')
        x = Conv2D(16, 3, activation='relu')(field)
        x = Conv2D(16, 2, activation='relu')(field)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = keras.layers.concatenate(
            [x, current_puyo, next_puyo, next_next_puyo])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(action_size, activation='linear', name='output')(x)
        model = Model(inputs=[field, current_puyo, next_puyo, next_next_puyo],
                      output=output)
        model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
        return model

    def calc_reward(self, step, reward, done):
        if done:
            return -1
        return reward / 15
        # if reward >= 15:
        #     return 1
        # else:
        #     return 0


if __name__ == "__main__":
    d = PuyoPuyoDQN(PuyoPuyo(), e_decay_rate=0.00001, memory_length=1000)
    d.run(
        num_episodes=50000,
        max_steps=200,
        model_save_episodes=1000,
        verbose=False)
