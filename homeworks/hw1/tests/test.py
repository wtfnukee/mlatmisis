import unittest
import numpy as np
import pandas as pd
import import_ipynb  # noqa: F401
from homework import (  # type: ignore
    border,
    checkerboard,
    sub,
    sort,
    animal_age,
    cats,
    mean_table,
    threesome,
)


class TestHomework1(unittest.TestCase):
    def test_border(self):
        expected = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        result = border()
        np.testing.assert_array_equal(result, expected)

    def test_checkerboard(self):
        expected = np.array(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ]
        )
        result = checkerboard()
        np.testing.assert_array_equal(result, expected)

    def test_sub(self):
        expected = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        result = sub()
        np.testing.assert_array_almost_equal(result, expected)

    def test_sort(self):
        expected = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
        result = sort()
        np.testing.assert_array_equal(result, expected)

    def setUp_pandas(self):
        self.data = {
            "animal": [
                "cat",
                "cat",
                "snake",
                "dog",
                "dog",
                "cat",
                "snake",
                "cat",
                "dog",
                "dog",
            ],
            "age": [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
            "visits": [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
            "priority": [
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "no",
                "yes",
                "no",
                "no",
            ],
        }
        self.labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        self.df = pd.DataFrame(self.data, index=self.labels)

    def test_animal_age(self):
        self.setUp_pandas()
        expected = pd.DataFrame(
            {"animal": ["snake", "dog", "dog"], "age": [0.5, np.nan, 5.0]},
            index=["c", "d", "e"],
        )
        result = animal_age(self.df)
        pd.testing.assert_frame_equal(result, expected)

    def test_cats(self):
        self.setUp_pandas()
        expected = pd.DataFrame(
            {
                "animal": ["cat", "cat"],
                "age": [2.5, 2.0],
                "visits": [1, 3],
                "priority": ["yes", "no"],
            },
            index=["a", "f"],
        )
        result = cats(self.df)
        pd.testing.assert_frame_equal(result, expected)

    def test_mean_table(self):
        self.setUp_pandas()
        result = mean_table(self.df)

        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.name, "animal")
        self.assertEqual(result.columns.name, "visits")

        self.assertEqual(set(result.index), {"cat", "dog", "snake"})
        self.assertEqual(set(result.columns), {1, 2, 3})

        self.assertAlmostEqual(result.loc["cat", 1], 2.5)
        self.assertTrue(pd.isna(result.loc["snake", 3]))
        self.assertAlmostEqual(result.loc["dog", 2], 6.0)

    def test_threesome(self):
        expected = pd.Series(
            [409, 200], index=pd.Index(["a", "b"], name="grps"), name="vals"
        )
        result = threesome()
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
