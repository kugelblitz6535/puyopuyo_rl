import gym
import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

from dqn import DQN


class GymDQN(DQN):
    def state2input(self, state):
        if isinstance(state, list):
            return np.array(state)
        else:
            return np.expand_dims(state, axis=0)

    def make_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=state_size.shape[0]))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(action_size.n, activation='linear'))

        # モデルのコンパイル
        model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
        return model

    def random_action(self):
        return self.env.action_space.sample()

    def illegal_actions(self):
        return []


if __name__ == "__main__":
    d = GymDQN(gym.make('CartPole-v0'))
    d.run(
        num_episodes=500,
        max_steps=200,
        batch_size=32,
        model_save_episodes=1000)
