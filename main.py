from collections import deque

import keras
import numpy as np
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

import puyopuyo

NUM_EPISODES = 50  # エピソード数
MAX_STEPS = 200  # 最大ステップ数
GAMMA = 0.99  # 時間割引率
WARMUP = 10  # 無操作ステップ数

E_START = 1.0  # εの初期値
E_STOP = 0.01  # εの最終値
E_DECAY_RATE = 0.001  # εの減衰率

MEMORY_SIZE = 10000  # 経験メモリのサイズ
BATCH_SIZE = 32  # バッチサイズ


class PuyoPuyoDQN(DQN):
    def __init__(self, env):
        self.env = env
        self.main_qn = self.make_model(
            self.env.observation_space, self.env.action_space)
        self.target_qn = self.make_model(
            self.env.observation_space, self.env.action_space)

    def __state2input(state):
    return [np.expand_dims(i, axis=0) for i in state]

    def make_model(self, state_size, action_size):
        field = Input(shape=state_size[0], name='field')
        next_puyo = Input(
            shape=state_size[1],
            name='next_puyo')
        next_next_puyo = Input(
            shape=state_size[2],
            name='next_next_puyo')

        x = Flatten()(field)
        x = Dense(128, activation='relu')(x)

        x = keras.layers.concatenate([x, next_puyo, next_next_puyo])
        output = Dense(action_size, activation='sigmoid', name='output')(x)
        model = Model(inputs=[field, next_puyo, next_next_puyo], output=output)
        model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
        return model

    def run(self):
        self.env = puyopuyo.PuyoPuyo()

        # 経験メモリの作成
        memory = Memory(MEMORY_SIZE)

        state = self.env.reset()

        # エピソード数分のエピソードを繰り返す
        total_step = 0  # 総ステップ数
        for episode in range(NUM_EPISODES):
            # target-networkの更新
            self.target_qn.set_weights(main_qn.get_weights())

            for step in range(MAX_STEPS):
                # εを減らす
                epsilon = E_STOP + (E_START - E_STOP) * \
                    np.exp(-E_DECAY_RATE * total_step)

                # ランダムな行動を選択
                if epsilon > np.random.rand():
                    action = np.random.choice(self.env.legal_actions())
                # 行動価値関数で行動を選択
                else:
                    ind = np.ones(self.env.action_space, dtype=bool)
                    ind[self.env.legal_actions()] = False
                    p = self.main_qn.predict(state2input(state))[0]
                    p[ind] = 0
                    action = np.argmax(p)

                # 行動に応じて状態と報酬を得る
                next_state, reward, done, _ = self.env.step(action)
                print(self.env.legal_actions())
                self.env.render()

                if step > WARMUP:
                    memory.add((state, action, reward, next_state))

                state = next_state

                # 行動価値関数の更新
                if len(memory) >= BATCH_SIZE:
                    # ニューラルネットワークの入力と出力の準備
                    inputs = [np.zeros((BATCH_SIZE, *shape))
                              for shape in self.env.observation_space]  # 入力(状態)
                    targets = np.zeros(
                        (BATCH_SIZE, self.env.action_space))  # 出力(行動ごとの価値)

                    # バッチサイズ分の経験をランダムに取得
                    minibatch = memory.sample(BATCH_SIZE)

                    # ニューラルネットワークの入力と出力の生成
                    for i, (state_b, action_b, reward_b,
                            next_state_b) in enumerate(minibatch):

                        # 入力に状態を指定
                        for j in range(len(self.env.observation_space)):
                            inputs[j][i] = state_b[j]

                        # 採った行動の価値を計算
                        if not done:
                            target = reward_b + GAMMA * \
                                np.amax(self.target_qn.predict(
                                    state2input(next_state_b))[0])
                        else:
                            target = reward_b

                        # 出力に行動ごとの価値を指定
                        targets[i] = self.main_qn.predict(state2input(state_b))
                        targets[i][action_b] = target  # 採った行動の価値

                    # 行動価値関数の更新
                    self.main_qn.fit(inputs, targets, epochs=1, verbose=0)

                if done:
                    print(
                        'エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(
                            episode, step, epsilon))
                    break
                total_step += 1

            state = self.env.reset()


class Memory():
    # 初期化
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    # 経験の追加
    def add(self, experience):
        self.buffer.append(experience)

    # バッチサイズ分の経験をランダムに取得
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    # 経験メモリのサイズ
    def __len__(self):
        return len(self.buffer)


def state2input(state):
    return [np.expand_dims(i, axis=0) for i in state]
