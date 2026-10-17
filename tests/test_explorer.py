import unittest

from explorer import run_experiment


class ExplorerTests(unittest.TestCase):
    def test_both_models_return_reproducible_metrics(self):
        for name in ("Logistic regression", "K-nearest neighbors"):
            result = run_experiment(name)
            self.assertEqual(result, run_experiment(name))
            self.assertEqual(len(result["confusion_matrix"]), 3)
            self.assertGreater(result["test_count"], 0)

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            run_experiment("unsupported")

    def test_invalid_test_fraction_raises(self):
        with self.assertRaises(ValueError):
            run_experiment(test_fraction=0.05)


if __name__ == "__main__":
    unittest.main()
