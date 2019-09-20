import numpy as np
import random


class DQN(object):
    def __init__(self, env):
        self.env = env
        self.main_qn = self.make_model(
            env.observation_space, env.action_space)
        self.target_qn = self.make_model(
            env.observation_space, env.action_space)
        self.memory = []

        self.gamma = 0.99  # 時間割引率

        self.e_start = 1.0  # εの初期値
        self.e_stop = 0.01  # εの最終値
        self.e_decay_rate = 0.001  # εの減衰率

    def make_model(self, state_size, action_size):
        raise NotImplementedError

    def state2input(self, state):
        raise NotImplementedError

    def random_action(self):
        return np.random.choice(self.env.legal_actions())

    def illegal_actions(self):
        ind = np.ones(self.env.action_space, dtype=bool)
        ind[self.env.legal_actions()] = False
        return ind

    def __calc_epsilon(self, total_step):
        return self.e_stop + (self.e_start - self.e_stop) * \
            np.exp(-self.e_decay_rate * total_step)

    def __update_evaluate_function(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # バッチサイズ分の経験をランダムに取得
        minibatch = random.sample(self.memory, batch_size)

        inputs = self.state2input([state for (state, _, _, _, _) in minibatch])
        targets = self.main_qn.predict(inputs)

        # ニューラルネットワークの入力と出力の生成
        for i, (_, action_b, reward_b,
                next_state_b, done) in enumerate(minibatch):
            # 採った行動の価値を計算
            if not done:
                target = reward_b + self.gamma * \
                    np.amax(self.target_qn.predict(
                        self.state2input(next_state_b))[0])
            else:
                target = reward_b

            targets[i][action_b] = target  # 採った行動の価値

        # 行動価値関数の更新
        self.main_qn.fit(inputs, targets, epochs=1, verbose=0)

    def run(
            self,
            num_episodes=50,
            max_steps=200,
            batch_size=32,
            model_save_episodes=10,
            verbose=True):
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
                    action = self.random_action()
                # 行動価値関数で行動を選択
                else:
                    p = self.main_qn.predict(self.state2input(state))[0]
                    ind = self.illegal_actions()
                    p[ind] = 0
                    action = np.argmax(p)

                # 行動に応じて状態と報酬を得る
                next_state, reward, done, _ = self.env.step(action)
                if 1 < reward:
                    print(reward)
                if verbose:
                    self.env.render()

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                self.__update_evaluate_function(batch_size)
                if episode % model_save_episodes == 0:
                    self.main_qn.save_weights(f"model/{episode}.hdf5")

                if done:
                    print(
                        'エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(
                            episode, step, epsilon))
                    break
                total_step += 1
