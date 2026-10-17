"""Small, reproducible classification experiments on scikit-learn Iris."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_experiment(model_name="Logistic regression", test_fraction=0.25, seed=7):
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    data = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        data.data,
        data.target,
        test_size=test_fraction,
        random_state=seed,
        stratify=data.target,
    )
    if model_name == "Logistic regression":
        estimator = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == "K-nearest neighbors":
        estimator = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("unsupported model")
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return {
        "accuracy": float(accuracy_score(test_y, prediction)),
        "confusion_matrix": confusion_matrix(test_y, prediction).tolist(),
        "labels": list(data.target_names),
        "features": list(data.feature_names),
        "test_count": len(test_y),
    }
