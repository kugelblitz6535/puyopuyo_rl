import unittest
from puyopuyo import PuyoPuyo


class TestPuyoPuyo(unittest.TestCase):

    def test_new_field(self):
        self.assertEqual(PuyoPuyo()._PuyoPuyo__new_field().shape, (11, 6))

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
