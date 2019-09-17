import numpy as np


class DQN(object):
    def __init__(self, env):
        self.env = env
        self.main_qn = self.__make_model()
        self.target_qn = self.__make_model()

        self.gamma = 0.99  # 時間割引率

        self.e_start = 1.0  # εの初期値
        self.e_stop = 0.01  # εの最終値
        self.e_decay_rate = 0.001  # εの減衰率

    def __make_model(self):
        raise NotImplementedError

    def __state2input(self, state):
        raise NotImplementedError

    def __calc_epsilon(self, total_step):
        return self.e_stop + (self.e_start - self.e_stop) * np.exp(-self.e_decay_rate * total_step)

    def __update_evaluate_function(self):
        if len(memory) >= batch_size:
            # ニューラルネットワークの入力と出力の準備
            inputs = [np.zeros((batch_size, *shape))
                      for shape in self.env.observation_space]  # 入力(状態)
            targets = np.zeros(
                (batch_size, self.env.action_space))  # 出力(行動ごとの価値)

            # バッチサイズ分の経験をランダムに取得
            minibatch = memory.sample(batch_size)

            # ニューラルネットワークの入力と出力の生成
            for i, (state_b, action_b, reward_b,
                    next_state_b) in enumerate(minibatch):

                # 入力に状態を指定
                for j in range(len(self.env.observation_space)):
                    inputs[j][i] = state_b[j]

                # 採った行動の価値を計算
                if not done:
                    target = reward_b + gamma * \
                        np.amax(self.target_qn.predict(
                            self.__state2input(next_state_b))[0])
                else:
                    target = reward_b

                # 出力に行動ごとの価値を指定
                targets[i] = self.main_qn.predict(
                    self.__state2input(state_b))
                targets[i][action_b] = target  # 採った行動の価値

            # 行動価値関数の更新
            self.main_qn.fit(inputs, targets, epochs=1, verbose=0)

    def run(self, num_episodes=50, max_steps=200, batch_size=32):
        # 経験メモリの作成
        memory_size = num_episodes * max_steps
        memory = Memory(memory_size)

        # エピソード数分のエピソードを繰り返す
        total_step = 0  # 総ステップ数
        for episode in range(num_episodes):
            state = self.env.reset()

            # target-networkの更新
            self.target_qn.set_weights(self.main_qn.get_weights())

            for step in range(max_steps):
                # ランダムな行動を選択
                epsilon = self.__calc_epsilon(total_step)
                if epsilon > np.random.rand():
                    action = np.random.choice(self.env.legal_actions())
                # 行動価値関数で行動を選択
                else:
                    ind = np.ones(self.env.action_space, dtype=bool)
                    ind[self.env.legal_actions()] = False
                    p = self.main_qn.predict(self.__state2input(state))[0]
                    p[ind] = 0
                    action = np.argmax(p)

                # 行動に応じて状態と報酬を得る
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()

                memory.add((state, action, reward, next_state))
                state = next_state

                self.__update_evaluate_function()

                if done:
                    print(
                        'エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(
                            episode, step, epsilon))
                    break
                total_step += 1


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
