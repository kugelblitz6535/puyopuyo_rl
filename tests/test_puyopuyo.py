import unittest

import numpy as np

from puyopuyo import PuyoPuyo


class TestPuyoPuyo(unittest.TestCase):

    def test_legal_actions(self):
        env = PuyoPuyo()
        np.testing.assert_array_equal(env.legal_actions(), np.arange(22))
        env.field = np.random.randint(1, 5, size=(11, 6))
        np.testing.assert_array_equal(env.legal_actions(), np.array([]))
        env.field[0] = 0
        env.field[1] = 0
        env.field[0, 1] = 1
        env.field[[0, 1, 1, 0, 1], [1, 1, 4, 5, 5]] = 1
        np.testing.assert_array_equal(env.legal_actions(), np.array(
            [7, 8, 9, 11, 12, 13, 14, 18]))

    def test_reset(self):
        env = PuyoPuyo()
        env.field = np.random.randint(1, 5, size=(11, 6))
        env.score = 100
        env.done = True
        env.reset()
        np.testing.assert_array_equal(
            env.field, np.zeros((11, 6), dtype=np.uint8))
        self.assertEqual(env.score, 0)
        self.assertFalse(env.done)

    def test_step(self):
        env = PuyoPuyo()
        field, next_puyo, next_next_puyo = env.reset()
        np.testing.assert_array_equal(field, np.zeros((11, 6), dtype=np.uint8))
        (field, new_next_puyo, _), _, done, _ = env.step(8)
        self.assertFalse(done)
        self.assertEqual(field[10, 2], next_puyo[0])
        self.assertEqual(field[10, 3], next_puyo[1])
        np.testing.assert_array_equal(new_next_puyo, next_next_puyo)

    def test_step_with_chain(self):
        env = PuyoPuyo()
        env.field = np.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]]
        )
        env.next_puyo = np.array([1, 1])
        (field, _, _), chain, done, _ = env.step(7)
        np.testing.assert_array_equal(field, np.zeros((11, 6), dtype=np.uint8))
        self.assertEqual(chain, 1)
        self.assertFalse(done)
        # env.field = np.array(
        #     [[0, 0, 0, 0, 0, 1],
        #      [0, 0, 0, 0, 1, 1],
        #      [2, 1, 3, 2, 3, 1],
        #      [3, 3, 1, 3, 2, 4],
        #      [2, 3, 1, 1, 4, 3],
        #      [2, 2, 1, 3, 3, 3],
        #      [3, 1, 3, 4, 3, 3],
        #      [1, 1, 2, 3, 4, 4],
        #      [3, 4, 2, 1, 1, 4],
        #      [3, 3, 4, 2, 2, 1],
        #      [4, 4, 2, 1, 1, 4]]
        # )


if __name__ == '__main__':
    unittest.main()
