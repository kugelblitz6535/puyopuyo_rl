import numpy as np


class PuyoPuyo(object):
    def __init__(self, width=6, height=10):
        self.width = width
        self.height = height
        self.observation_space = ((self.height + 1, self.width), 2, 2)
        self.action_space = self.width * 4 - 2
        self.field = np.zeros(self.observation_space[0])
        self.next_puyo = self.__get_next_puyo()
        self.next_next_puyo = self.__get_next_puyo()
        self.colors = 4
        self.reward = 0
        self.done = False

        self.regal_actions = self.__list_regal_actions()

    def __get_next_puyo(self):
        return np.random.randint(self.colors + 1, size=2)

    def __list_regal_actions(self):
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
        self.next_puyo = self.next_next_puyo
        self.next_next_puyo = self.__get_next_puyo()
        info = None
        return (self.field, self.next_puyo,
                self.next_next_puyo), self.reward, self.done, info

    def put(self, action):
        pass

    def erase(self):
        pass

    def render(self):
        print(self.field)
