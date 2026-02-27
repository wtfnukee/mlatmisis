import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import import_ipynb  # noqa: F401
from homework import OVAClassifier, AVAClassifier, LARS  # type: ignore


class TestOVA(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=6,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        self.X_train, self.X_test = self.X[:300], self.X[300:]
        self.y_train, self.y_test = self.y[:300], self.y[300:]

    def test_ova_fit_predict(self):
        clf = OVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertEqual(preds.shape, self.y_test.shape)

    def test_ova_accuracy(self):
        clf = OVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        self.assertGreater(acc, 0.7)

    def test_ova_all_classes_present(self):
        clf = OVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertEqual(set(preds), set(self.y_test))


class TestAVA(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=6,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        self.X_train, self.X_test = self.X[:300], self.X[300:]
        self.y_train, self.y_test = self.y[:300], self.y[300:]

    def test_ava_fit_predict(self):
        clf = AVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertEqual(preds.shape, self.y_test.shape)

    def test_ava_accuracy(self):
        clf = AVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        self.assertGreater(acc, 0.7)

    def test_ava_all_classes_present(self):
        clf = AVAClassifier(LogisticRegression(max_iter=1000))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertEqual(set(preds), set(self.y_test))


class TestLARS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=3,
            noise=5.0,
            random_state=42,
        )
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.y = self.y - self.y.mean()

    def test_lars_fit_predict(self):
        model = LARS()
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_lars_coef_shape(self):
        model = LARS()
        model.fit(self.X, self.y)
        self.assertEqual(model.coef_.shape[0], self.X.shape[1])

    def test_lars_coef_path(self):
        model = LARS()
        model.fit(self.X, self.y)
        self.assertGreater(len(model.coef_path_), 1)
        first_step = model.coef_path_[0]
        n_nonzero = np.count_nonzero(np.abs(first_step) > 1e-10)
        self.assertLessEqual(n_nonzero, 2)

    def test_lars_n_features(self):
        model = LARS(n_features=3)
        model.fit(self.X, self.y)
        n_nonzero = np.count_nonzero(np.abs(model.coef_) > 1e-10)
        self.assertLessEqual(n_nonzero, 3)

    def test_lars_reasonable_predictions(self):
        from sklearn.linear_model import Lars as SkLars

        model = LARS()
        model.fit(self.X, self.y)
        sk_model = SkLars().fit(self.X, self.y)
        our_mse = np.mean((model.predict(self.X) - self.y) ** 2)
        sk_mse = np.mean((sk_model.predict(self.X) - self.y) ** 2)
        self.assertLess(our_mse, sk_mse * 3)


if __name__ == "__main__":
    unittest.main()
