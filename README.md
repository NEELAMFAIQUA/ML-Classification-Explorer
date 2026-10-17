# Classification Explorer

An original hands-on classification comparison using scikit-learn's built-in
Iris dataset, stratified train/test split, standardization, Logistic Regression
or K-nearest neighbors, accuracy, and a confusion matrix.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Metrics depend on the sample split and are for learning only, not general
performance guarantees. Run tests with `python -m unittest discover -s tests -v`.
