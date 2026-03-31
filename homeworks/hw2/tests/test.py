import unittest
import numpy as np
import time
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import import_ipynb  # noqa: F401
from homework import (  # type: ignore
    GradientBoostingMSE,
    prepare_catboost_pool,
    train_with_early_stopping,
    compare_boosters,
)


class TestGradientBoostingMSE(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_regression(
            n_samples=500, n_features=10, n_informative=5,
            noise=10.0, random_state=42,
        )
        self.X_train, self.X_test = self.X[:400], self.X[400:]
        self.y_train, self.y_test = self.y[:400], self.y[400:]

    def test_fit_predict_shape(self):
        model = GradientBoostingMSE(n_estimators=10, max_depth=3, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(preds.shape, self.y_test.shape)

    def test_better_than_mean(self):
        model = GradientBoostingMSE(n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        our_mse = mean_squared_error(self.y_test, preds)
        mean_mse = mean_squared_error(self.y_test, np.full_like(self.y_test, self.y_train.mean()))
        self.assertLess(our_mse, mean_mse)

    def test_close_to_sklearn(self):
        model = GradientBoostingMSE(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        our_mse = mean_squared_error(self.y_test, model.predict(self.X_test))

        sk = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        sk.fit(self.X_train, self.y_train)
        sk_mse = mean_squared_error(self.y_test, sk.predict(self.X_test))

        self.assertLess(our_mse, sk_mse * 3)

    def test_learning_rate_effect(self):
        mse_fast = self._train_and_score(learning_rate=0.5, n_estimators=20)
        mse_slow = self._train_and_score(learning_rate=0.01, n_estimators=20)
        # With few iterations, higher lr should fit better (lower MSE)
        self.assertLess(mse_fast, mse_slow)

    def test_more_trees_improve(self):
        mse_few = self._train_and_score(n_estimators=5)
        mse_many = self._train_and_score(n_estimators=100)
        self.assertLess(mse_many, mse_few)

    def _train_and_score(self, n_estimators=50, max_depth=3, learning_rate=0.1):
        model = GradientBoostingMSE(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        )
        model.fit(self.X_train, self.y_train)
        return mean_squared_error(self.y_test, model.predict(self.X_test))


class TestCatBoostPractice(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=1000, n_features=10, n_informative=6,
            n_classes=2, random_state=42,
        )
        self.X_train, self.X_test = self.X[:800], self.X[800:]
        self.y_train, self.y_test = self.y[:800], self.y[800:]
        self.X_val, self.y_val = self.X[600:800], self.y[600:800]
        self.X_train_small = self.X[:600]
        self.y_train_small = self.y[:600]

    def test_prepare_pool(self):
        from catboost import Pool
        train_pool, val_pool = prepare_catboost_pool(
            self.X_train_small, self.y_train_small,
            self.X_val, self.y_val,
            cat_features=None,
        )
        self.assertIsInstance(train_pool, Pool)
        self.assertIsInstance(val_pool, Pool)

    def test_early_stopping(self):
        from catboost import Pool
        train_pool = Pool(self.X_train_small, label=self.y_train_small)
        val_pool = Pool(self.X_val, label=self.y_val)
        model, best_iter = train_with_early_stopping(
            train_pool, val_pool, iterations=500, patience=30,
        )
        self.assertLess(best_iter, 500)
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        self.assertGreater(acc, 0.7)

    def test_compare_boosters_format(self):
        results = compare_boosters(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
        )
        self.assertIsInstance(results, dict)
        for name in ['CatBoost', 'XGBoost', 'LightGBM']:
            self.assertIn(name, results)
            self.assertIn('accuracy', results[name])
            self.assertIn('time', results[name])

    def test_compare_boosters_accuracy(self):
        results = compare_boosters(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
        )
        for name, res in results.items():
            self.assertGreater(res['accuracy'], 0.7, f'{name} accuracy too low')


if __name__ == "__main__":
    unittest.main()
