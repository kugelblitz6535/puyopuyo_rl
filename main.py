from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

import puyopuyo


NUM_EPISODES = 500  # エピソード数
MAX_STEPS = 2000  # 最大ステップ数
GAMMA = 0.99  # 時間割引率
WARMUP = 10  # 無操作ステップ数

E_START = 1.0  # εの初期値
E_STOP = 0.01  # εの最終値
E_DECAY_RATE = 0.001  # εの減衰率

MEMORY_SIZE = 10000  # 経験メモリのサイズ
BATCH_SIZE = 32  # バッチサイズ


def make_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=state_size))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
    return model


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


env = puyopuyo.PuyoPuyo()

env.reset()

field, next_puyo, next_next_puyo = env.observation_space
action_size = env.action_space

# main-networkの作成
main_qn = make_model(state_size, action_size)

# target-networkの作成
target_qn = make_model(state_size, action_size)

# 経験メモリの作成
memory = Memory(MEMORY_SIZE)
# 学習の開始

# 環境の初期化
field, next_puyo, next_next_puyo = env.reset()

# エピソード数分のエピソードを繰り返す
total_step = 0  # 総ステップ数
success_count = 0  # 成功数
for episode in range(1, NUM_EPISODES + 1):
    step = 0  # ステップ数

    # target-networkの更新
    target_qn.set_weights(main_qn.get_weights())

    # 1エピソードのループ
    for _ in range(1, MAX_STEPS + 1):
        step += 1
        total_step += 1

        # εを減らす
        epsilon = E_STOP + (E_START - E_STOP) * \
            np.exp(-E_DECAY_RATE * total_step)

        # ランダムな行動を選択
        if epsilon > np.random.rand():
            action = env.action_space.sample()
        # 行動価値関数で行動を選択
        else:
            action = np.argmax(main_qn.predict(state)[0])

        # 行動に応じて状態と報酬を得る
        next_state, _, done, _ = env.step(action)
        env.render()
        next_state = np.reshape(next_state, [1, state_size])

        # エピソード完了時
        if done:
            # 報酬の指定
            if step >= MAX_STEPS * 0.9:
                success_count += 1
                reward = 1
            else:
                success_count = 0
                reward = 0

            # 次の状態に状態なしを代入
            next_state = np.zeros(state.shape)

            # 経験の追加
            if step > WARMUP:
                memory.add((state, action, reward, next_state))
        # エピソード完了でない時
        else:
            # 報酬の指定
            reward = 0

            # 経験の追加
            if step > WARMUP:
                memory.add((state, action, reward, next_state))

            # 状態に次の状態を代入
            state = next_state

        # 行動価値関数の更新
        if len(memory) >= BATCH_SIZE:
            # ニューラルネットワークの入力と出力の準備
            inputs = np.zeros((BATCH_SIZE, state_size))  # 入力(状態)
            targets = np.zeros((BATCH_SIZE, action_size))  # 出力(行動ごとの価値)

            # バッチサイズ分の経験をランダムに取得
            minibatch = memory.sample(BATCH_SIZE)

            # ニューラルネットワークの入力と出力の生成
            for i, (state_b, action_b, reward_b,
                    next_state_b) in enumerate(minibatch):

                # 入力に状態を指定
                inputs[i] = state_b

                # 採った行動の価値を計算
                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    target = reward_b + GAMMA * \
                        np.amax(target_qn.predict(next_state_b)[0])
                else:
                    target = reward_b

                # 出力に行動ごとの価値を指定
                targets[i] = main_qn.predict(state_b)
                targets[i][action_b] = target  # 採った行動の価値

            # 行動価値関数の更新
            main_qn.fit(inputs, targets, epochs=1, verbose=0)

        # エピソード完了時
        if done:
            # エピソードループを抜ける
            break

    # エピソード完了時のログ表示
    print('エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(episode, step, epsilon))

    # 5回連続成功で学習終了
    if success_count >= 5:
        break

    # 環境のリセット
    state = env.reset()
