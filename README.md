# Classification Explorer

An original interactive ML learning app with two focused examples: Logistic
Regression for ad-click classification and K-nearest neighbors for finding
similar fictional Titanic-shaped passenger profiles.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The datasets are generated teaching examples, not actual advertising outcomes
or historical passenger records. Metrics and distances are for learning only,
not general performance guarantees. Run tests with
`python -m unittest discover -s tests -v`.
