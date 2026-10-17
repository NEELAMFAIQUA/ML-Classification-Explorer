import unittest

from explorer import (
    ad_click_data,
    ad_click_experiment,
    find_similar_passengers,
    synthetic_passengers,
)


class ExplorerTests(unittest.TestCase):
    def test_ad_click_dataset_and_logistic_experiment(self):
        data = ad_click_data()
        self.assertEqual(list(data.columns), ["Age", "EstimatedSalary", "Clicked"])
        result = ad_click_experiment()
        self.assertEqual(result["test_count"], 125)
        self.assertGreaterEqual(result["accuracy"], 0)
        self.assertLessEqual(result["accuracy"], 1)

    def test_invalid_test_fraction_raises(self):
        with self.assertRaises(ValueError):
            ad_click_experiment(0.05)

    def test_knn_returns_nearest_distinct_profiles(self):
        passengers = synthetic_passengers()
        result = find_similar_passengers(passengers, 0, 5)
        self.assertEqual(len(result), 5)
        self.assertNotIn(0, result.index)
        self.assertIn("Distance", result.columns)

    def test_knn_validates_passenger_and_count(self):
        passengers = synthetic_passengers()
        with self.assertRaises(ValueError):
            find_similar_passengers(passengers, 999)
        with self.assertRaises(ValueError):
            find_similar_passengers(passengers, 0, 0)


if __name__ == "__main__":
    unittest.main()
