import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lars as SklearnLars
import import_ipynb  # noqa: F401
from homework import LARS  # type: ignore


def test_homework_lars():
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 3, 5, 7, 11])

    model = LARS()
    model.fit(X_train, y_train)
    preds = model.predict(X_train)

    assert preds.shape == (5,), "Predictions should match number of samples"
    assert r2_score(y_train, preds) > 0.8, "R2 should be above 0.8"

    sklearn_model = SklearnLars()
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_train)

    assert np.allclose(preds, sklearn_preds, rtol=1e-03), (
        "Predictions should match sklearn's implementation"
    )
