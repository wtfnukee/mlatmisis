import unittest
from homeworks.hw0.task import hello


class TestHello(unittest.TestCase):
    def test_main_returns_hello_world(self):
        expected = "Привет, мир!"
        result = hello()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
