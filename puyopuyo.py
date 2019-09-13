import numpy as np


class PuyoPuyo(object):
    def __init__(self, width=6, height=10):
        self.width = width
        self.height = height
        self.colors = 4
        self.observation_space = ((self.height + 1, self.width), (2,), (2,))
        self.action_space = self.width * 4 - 2
        self.field = np.zeros(self.observation_space[0])
        self.next_puyo = self.__get_next_puyo()
        self.next_next_puyo = self.__get_next_puyo()
        self.reward = 0
        self.done = False
        actions = [(i, j)for i in range(
            self.width) for j in range(4)]
        actions.pop(3)
        actions.pop(-3)
        self.actions = actions

    def __get_next_puyo(self):
        return np.random.randint(self.colors + 1, size=2)

    def legal_actions(self):
        return np.arange(self.action_space)

    def reset(self):
        self.field = np.zeros(self.observation_space[0])
        self.next_puyo = self.__get_next_puyo()
        self.next_next_puyo = self.__get_next_puyo()
        self.reward = 0
        self.done = False
        return self.field, self.next_puyo, self.next_next_puyo

    def step(self, action):
        self.put(action)
        self.erase()
        if not self.field[1][2] == 0:
            self.done = True
        self.next_puyo = self.next_next_puyo
        self.next_next_puyo = self.__get_next_puyo()
        info = None
        return (self.field, self.next_puyo,
                self.next_next_puyo), self.reward, self.done, info

    def drop(self, col, puyopuyo):
        i = np.argmax(np.where(self.field[:, col] == 0))

        if isinstance(puyopuyo, np.int64):
            puyopuyo = [puyopuyo]
        for puyo in puyopuyo:
            self.field[:, col][i] = puyo
            i -= 1

    def put(self, action):
        col, rotate = self.actions[action]
        if rotate == 0:
            self.drop(col, self.next_puyo)
        elif rotate == 2:
            self.drop(col, self.next_puyo[::-1])
        elif rotate == 1:
            self.drop(col, self.next_puyo[0])
            self.drop(col + 1, self.next_puyo[1])
        elif rotate == 3:
            self.drop(col, self.next_puyo[0])
            self.drop(col - 1, self.next_puyo[1])

    def erase(self):
        return False, 0

    def chain(self):
        success, point = self.erase()
        chain = 0
        score = 0
        while success:
            chain += 1
            score += point * chain
            success, point = self.erase()

        return chain, score

    def render(self):
        print(self.field)
