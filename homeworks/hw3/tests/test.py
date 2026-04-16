import unittest
import numpy as np
from sklearn.datasets import make_classification, load_iris, load_digits
from sklearn.decomposition import PCA as SkPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import import_ipynb  # noqa: F401
from homework import KNNClassifier, PCA, experiment_knn_pca  # type: ignore


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=300, n_features=10, n_informative=6,
            n_classes=3, n_clusters_per_class=1, random_state=42,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42,
        )

    def test_fit_predict_shape(self):
        clf = KNNClassifier(n_neighbors=5)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertEqual(preds.shape, self.y_test.shape)

    def test_accuracy_reasonable(self):
        clf = KNNClassifier(n_neighbors=5)
        clf.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, clf.predict(self.X_test))
        self.assertGreater(acc, 0.7)

    def test_classes_present(self):
        clf = KNNClassifier(n_neighbors=5)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.assertTrue(set(preds).issubset(set(self.y_train)))

    def test_k_equals_1_memorizes_train(self):
        """K=1 should perfectly fit the training data (each point is its own nearest neighbor)."""
        clf = KNNClassifier(n_neighbors=1)
        clf.fit(self.X_train, self.y_train)
        train_preds = clf.predict(self.X_train)
        self.assertEqual(accuracy_score(self.y_train, train_preds), 1.0)

    def test_different_k_different_results(self):
        """Different K should produce different predictions on some points."""
        preds_k1 = KNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train).predict(self.X_test)
        preds_k20 = KNNClassifier(n_neighbors=20).fit(self.X_train, self.y_train).predict(self.X_test)
        self.assertFalse(np.array_equal(preds_k1, preds_k20))

    def test_close_to_sklearn(self):
        """Our KNN should give very similar results to sklearn for same K."""
        our = KNNClassifier(n_neighbors=5)
        our.fit(self.X_train, self.y_train)
        our_preds = our.predict(self.X_test)

        sk = KNeighborsClassifier(n_neighbors=5, algorithm='brute').fit(self.X_train, self.y_train)
        sk_preds = sk.predict(self.X_test)

        # Allow small differences due to tie-breaking
        agreement = np.mean(our_preds == sk_preds)
        self.assertGreater(agreement, 0.9)


class TestPCA(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.n_samples, self.n_features = self.X.shape

    def test_transform_shape(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        X_transformed = pca.transform(self.X)
        self.assertEqual(X_transformed.shape, (self.n_samples, 2))

    def test_components_shape(self):
        pca = PCA(n_components=3)
        pca.fit(self.X)
        self.assertEqual(pca.components_.shape, (3, self.n_features))

    def test_explained_variance_sums_to_one(self):
        pca = PCA(n_components=self.n_features)
        pca.fit(self.X)
        self.assertAlmostEqual(pca.explained_variance_ratio_.sum(), 1.0, places=5)

    def test_explained_variance_sorted_descending(self):
        pca = PCA(n_components=self.n_features)
        pca.fit(self.X)
        evr = pca.explained_variance_ratio_
        self.assertTrue(np.all(np.diff(evr) <= 1e-10),
                        f"Variance ratios should be non-increasing, got {evr}")

    def test_first_component_matches_sklearn(self):
        """First component should explain the most variance (match sklearn up to sign)."""
        our = PCA(n_components=1).fit(self.X)
        sk = SkPCA(n_components=1).fit(self.X)
        our_var_ratio = our.explained_variance_ratio_[0]
        sk_var_ratio = sk.explained_variance_ratio_[0]
        self.assertAlmostEqual(our_var_ratio, sk_var_ratio, places=4)

    def test_inverse_transform(self):
        """Inverse transform should approximately reconstruct the input when all components kept."""
        pca = PCA(n_components=self.n_features)
        pca.fit(self.X)
        X_reduced = pca.transform(self.X)
        X_restored = pca.inverse_transform(X_reduced)
        np.testing.assert_allclose(X_restored, self.X, atol=1e-6)

    def test_reconstruction_error_decreases(self):
        """More components = smaller reconstruction error."""
        errors = []
        for k in [1, 2, 3, 4]:
            pca = PCA(n_components=k).fit(self.X)
            X_restored = pca.inverse_transform(pca.transform(self.X))
            errors.append(np.mean((self.X - X_restored) ** 2))
        self.assertTrue(all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1)))


class TestExperiment(unittest.TestCase):
    def setUp(self):
        digits = load_digits()
        self.X, self.y = digits.data, digits.target

    def test_experiment_returns_dict(self):
        results = experiment_knn_pca(self.X, self.y, n_components_list=[5, 10, 20])
        self.assertIsInstance(results, dict)

    def test_experiment_has_required_keys(self):
        results = experiment_knn_pca(self.X, self.y, n_components_list=[5, 10, 20])
        self.assertIn('n_components', results)
        self.assertIn('accuracies', results)
        self.assertEqual(len(results['n_components']), 3)
        self.assertEqual(len(results['accuracies']), 3)

    def test_experiment_accuracies_reasonable(self):
        """On digits, KNN with PCA should achieve reasonable accuracy."""
        results = experiment_knn_pca(self.X, self.y, n_components_list=[10, 30])
        for acc in results['accuracies']:
            self.assertGreater(acc, 0.7)


if __name__ == "__main__":
    unittest.main()
