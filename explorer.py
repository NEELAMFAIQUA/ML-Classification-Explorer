"""Two original classification lessons: ad clicks and passenger similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ad_click_data(count=500, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 66, count)
    salary = rng.integers(20000, 150001, count)
    logits = -2.8 + (age - 35) / 18 + (salary - 70000) / 65000
    probability = 1 / (1 + np.exp(-logits))
    clicks = rng.binomial(1, probability)
    return pd.DataFrame(
        {"Age": age, "EstimatedSalary": salary, "Clicked": clicks}
    )


def ad_click_experiment(test_fraction=0.25, seed=7):
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    data = ad_click_data(seed=seed)
    train_x, test_x, train_y, test_y = train_test_split(
        data[["Age", "EstimatedSalary"]],
        data["Clicked"],
        test_size=test_fraction,
        random_state=seed,
        stratify=data["Clicked"],
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=seed, max_iter=500)
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    return {
        "accuracy": float(accuracy_score(test_y, predicted)),
        "confusion_matrix": confusion_matrix(test_y, predicted).tolist(),
        "train_count": len(train_y),
        "test_count": len(test_y),
        "data": data,
    }


def synthetic_passengers(count=240, seed=11):
    """Create fictional passenger-shaped examples; not historical Titanic data."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["female", "male"], count, p=[0.42, 0.58])
    passenger_class = rng.choice([1, 2, 3], count, p=[0.22, 0.25, 0.53])
    age = rng.normal(30, 14, count).clip(1, 75).round(1)
    family_size = rng.integers(1, 6, count)
    fare = np.select(
        [passenger_class == 1, passenger_class == 2],
        [rng.normal(65, 18, count), rng.normal(24, 8, count)],
        default=rng.normal(10, 4, count),
    ).clip(3, 140).round(2)
    logits = (
        -1.2 + (sex == "female") * 1.8
        + (passenger_class == 1) * 0.65
        + (age < 14) * 0.75
        - (family_size > 4) * 0.35
    )
    survival_probability = 1 / (1 + np.exp(-logits))
    survived = rng.binomial(1, survival_probability)
    return pd.DataFrame(
        {
            "PassengerClass": passenger_class,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "FamilySize": family_size,
            "Survived": survived,
        }
    )


def find_similar_passengers(data, passenger_index, neighbors=5):
    feature_columns = ["PassengerClass", "Sex", "Age", "Fare", "FamilySize"]
    if passenger_index not in data.index:
        raise ValueError("Passenger index is not in the dataset")
    if not 1 <= neighbors < len(data):
        raise ValueError("neighbors must be positive and less than the row count")
    categorical = ["Sex"]
    numeric = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    matrix = transformer.fit_transform(data[feature_columns])
    model = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    model.fit(matrix)
    row_position = data.index.get_loc(passenger_index)
    distances, positions = model.kneighbors(
        matrix[row_position:row_position + 1], n_neighbors=neighbors + 1
    )
    selected = [
        (int(position), float(distance))
        for position, distance in zip(positions[0], distances[0])
        if int(position) != row_position
    ][:neighbors]
    results = data.iloc[[position for position, _ in selected]].copy()
    results.insert(0, "Distance", [distance for _, distance in selected])
    return results
