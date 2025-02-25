import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import import_ipynb  # noqa: F401
from homework import (  # type: ignore
    OneVsAllClassifier,
    AllVsAllClassifier,
)


def test_homework_multiclass():
    X_test = np.array([[1, 2], [4, 5], [7, 8], [2, 3], [5, 6]])
    y_test = np.array([0, 1, 2, 0, 1])

    # Test One-vs-All
    ova_classifier = OneVsAllClassifier()
    ova_classifier.fit(X_test, y_test)
    ova_preds = ova_classifier.predict(X_test)

    assert len(ova_preds) == len(X_test), "Predictions should match number of samples"
    assert accuracy_score(y_test, ova_preds) > 0.8, "Accuracy should be above 0.8"

    # Test All-vs-All
    ava_classifier = AllVsAllClassifier()
    ava_classifier.fit(X_test, y_test)
    ava_preds = ava_classifier.predict(X_test)

    assert len(ava_preds) == len(X_test), "Predictions should match number of samples"
    assert accuracy_score(y_test, ava_preds) > 0.8, "Accuracy should be above 0.8"
