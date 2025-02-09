import unittest
import import_ipynb  # noqa: F401
from homework import hello  # type: ignore


class TestHomework0(unittest.TestCase):
    def test_main_returns_hello_world(self):
        expected = "Привет, мир!"
        result = hello()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
