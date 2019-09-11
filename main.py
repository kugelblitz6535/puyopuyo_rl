from gym.envs.registration import make

from gym_puyopuyo.agent import TsuTreeSearchAgent
from gym_puyopuyo import register

register()

agent = TsuTreeSearchAgent()

env = make("PuyoPuyoEndlessTsu-v2")

env.reset()
state = env.get_root()

for i in range(30):
    action = agent.get_action(state)
    _, _, done, info = env.step(action)
    state = info["state"]
    if done:
        break

env.render()
