import numpy as np

# TODO: current puyo


class PuyoPuyo(object):
    def __init__(self, width=6, height=11):
        self.width = width
        self.height = height
        self.colors = 4
        self.erase_thresh = 4
        self.observation_space = ((self.height, self.width), (2,), (2,))
        self.action_space = self.width * 4 - 2
        self.field = self.__new_field()
        self.visited = np.zeros(self.observation_space[0], dtype=np.bool)
        self.erase_map = np.zeros(self.observation_space[0], dtype=np.bool)
        self.next_puyo = self.__get_next_puyo()
        self.next_next_puyo = self.__get_next_puyo()
        self.score = 0
        self.done = False
        actions = [(i, j)for i in range(
            self.width) for j in range(4)]
        actions.pop(3)
        actions.pop(-3)
        self.actions = actions

    def __new_field(self):
        return np.zeros(self.observation_space[0], dtype=np.uint8)

    def __get_next_puyo(self):
        return np.random.randint(1, self.colors + 1, size=2)

    def legal_actions(self):
        """選択可能な手の配列を返す。埋まってる列の上には置けない。上まで埋まってる列の向こうには置けない。
        """
        legal = np.zeros(len(self.actions), dtype=np.bool)
        puttable = self.field[0] == 0
        for i in range(self.action_space):
            col, rotate = self.actions[i]
            if col < 2:
                if not np.all(puttable[col:2]):
                    continue
            elif 2 < col:
                if not np.all(puttable[3:col+1]):
                    continue
            if rotate == 0 or rotate == 2:
                if self.field[:, col][1] == 0:
                    legal[i] = True
            elif rotate == 1:
                if puttable[col] and puttable[col+1]:
                    legal[i] = True
            elif rotate == 3:
                if puttable[col] and puttable[col-1]:
                    legal[i] = True

        return np.where(legal)[0]

    def reset(self):
        """環境をリセットする。
        """
        self.field = self.__new_field()
        self.next_puyo = self.__get_next_puyo()
        self.next_next_puyo = self.__get_next_puyo()
        self.score = 0
        self.done = False
        return self.field, self.next_puyo, self.next_next_puyo

    def step(self, action):
        """actionで指定した手を行った後の環境、報酬、環境の終了状態、infoを返す。
        """
        self.__put(action)
        chain, point = self.__chain()
        self.score += point
        if not self.field[1][2] == 0:
            self.done = True
        self.next_puyo = self.next_next_puyo
        self.next_next_puyo = self.__get_next_puyo()
        info = None
        return (self.field, self.next_puyo,
                self.next_next_puyo), chain, self.done, info

    def __drop(self, col, puyopuyo):
        i = np.argmax(np.where(self.field[:, col] == 0))

        if isinstance(puyopuyo, np.int64):
            puyopuyo = [puyopuyo]
        for puyo in puyopuyo:
            self.field[:, col][i] = puyo
            i -= 1

    def __put(self, action):
        col, rotate = self.actions[action]
        if rotate == 0:
            self.__drop(col, self.next_puyo)
        elif rotate == 2:
            self.__drop(col, self.next_puyo[::-1])
        elif rotate == 1:
            self.__drop(col, self.next_puyo[0])
            self.__drop(col + 1, self.next_puyo[1])
        elif rotate == 3:
            self.__drop(col, self.next_puyo[0])
            self.__drop(col - 1, self.next_puyo[1])

    def __out_of_field(self, row, col):
        return row < 0 or self.height <= row or col < 0 or self.width <= col

    def __check(self, row, col, color):
        if self.__out_of_field(row, col):
            return 0
        if self.visited[row][col]:
            return 0
        if self.field[row][col] == 0:
            return 0
        if self.field[row][col] != color:
            return 0

        self.visited[row][col] = True
        self.erase_map[row][col] = True
        return 1 + \
            self.__check(row-1, col, color) + \
            self.__check(row, col-1, color) + \
            self.__check(row+1, col, color) + \
            self.__check(row, col+1, color)

    def __fall(self):
        for col in range(self.width):
            exist = np.where(self.field[:, col] != 0)[0]
            if exist.size == 0:
                continue
            fell = np.zeros(self.height, dtype=np.uint8)
            fell[-exist.size:] = self.field[exist, col]
            self.field[:, col] = fell

    def __erase(self):
        self.visited = np.zeros(self.observation_space[0], dtype=np.bool)
        point = 0
        for row in range(self.height):
            for col in range(self.width):
                self.erase_map = np.zeros(
                    self.observation_space[0], dtype=np.bool)
                n = self.__check(row, col, self.field[row][col])
                if self.erase_thresh <= n:
                    point += n * 10
                    self.field[self.erase_map] = 0
                    self.__fall()

        if 0 < point:
            return True, point
        else:
            return False, 0

    def __chain(self):
        success, got_point = self.__erase()
        chain = 0
        point = 0
        while success:
            chain += 1
            point += got_point * chain
            success, got_point = self.__erase()

        return chain, point

    def render(self):
        for row in self.field:
            for puyo in row:
                if puyo == 0:
                    print('\x1b[0m\u00b7', end="")
                else:
                    print(self.__int2puyo(puyo), end="")
                print(' ', end="")
            print('\x1b[0m')
        print()

    @staticmethod
    def __int2puyo(n):
        return f"\x1b[3{n}m\u25cf"


if __name__ == "__main__":
    env = PuyoPuyo()
    done = False
    while not done:
        action = np.random.choice(env.legal_actions())
        next_state, reward, done, _ = env.step(action)
        env.render()
