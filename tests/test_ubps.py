"""
Unit tests for User Behaviour Prediction System.
Run with: pytest tests/
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataset mimicking UCI Online Shoppers structure."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'Administrative':        np.random.randint(0, 10, n),
        'Administrative_Duration': np.random.uniform(0, 100, n),
        'Informational':         np.random.randint(0, 5, n),
        'Informational_Duration': np.random.uniform(0, 50, n),
        'ProductRelated':        np.random.randint(0, 30, n),
        'ProductRelated_Duration': np.random.uniform(0, 500, n),
        'BounceRates':           np.random.uniform(0, 0.2, n),
        'ExitRates':             np.random.uniform(0, 0.2, n),
        'PageValues':            np.random.uniform(0, 50, n),
        'SpecialDay':            np.random.uniform(0, 1, n),
        'Month':                 np.random.choice(['Feb','Mar','May','Jun','Nov'], n),
        'OperatingSystems':      np.random.randint(1, 6, n),
        'Browser':               np.random.randint(1, 8, n),
        'Region':                np.random.randint(1, 9, n),
        'TrafficType':           np.random.randint(1, 10, n),
        'VisitorType':           np.random.choice(['Returning_Visitor','New_Visitor','Other'], n),
        'Weekend':               np.random.choice([True, False], n),
        'Revenue':               np.random.choice([True, False], n, p=[0.155, 0.845]),
    })


# ── Data tests ────────────────────────────────────────────────────────────────

def test_dataset_columns(sample_df):
    """Dataset must have all expected 18 columns."""
    expected = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend', 'Revenue'
    ]
    for col in expected:
        assert col in sample_df.columns, f"Missing column: {col}"


def test_target_is_binary(sample_df):
    """Revenue column must be binary (True/False or 0/1)."""
    unique_vals = set(sample_df['Revenue'].unique())
    assert unique_vals.issubset({True, False, 0, 1}), \
        f"Revenue has unexpected values: {unique_vals}"


def test_no_negative_durations(sample_df):
    """Duration columns must be non-negative."""
    for col in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']:
        assert (sample_df[col] >= 0).all(), f"{col} has negative values"


# ── Preprocessing tests ───────────────────────────────────────────────────────

def test_encoding_removes_categoricals(sample_df):
    """One-hot encoding must remove all original categorical columns."""
    from sklearn.preprocessing import LabelEncoder

    cat_cols = ['Month', 'VisitorType', 'Browser', 'Region', 'TrafficType', 'OperatingSystems']
    X = sample_df.drop('Revenue', axis=1)
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    for col in cat_cols:
        assert col not in X_enc.columns, f"Categorical column still present: {col}"


def test_train_test_split_ratio(sample_df):
    """80/20 split must produce correct proportions."""
    from sklearn.model_selection import train_test_split

    X = sample_df.drop('Revenue', axis=1)
    y = sample_df['Revenue'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    total = len(sample_df)
    assert abs(len(X_train) / total - 0.80) < 0.02, "Train split not ~80%"
    assert abs(len(X_test)  / total - 0.20) < 0.02, "Test split not ~20%"


def test_scaler_output_range(sample_df):
    """StandardScaler output must have mean ≈ 0 and std ≈ 1."""
    from sklearn.preprocessing import StandardScaler

    numeric = sample_df.select_dtypes(include=[np.number]).drop(columns=['Revenue'], errors='ignore')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    assert abs(scaled.mean()) < 0.1, "Scaled mean not near 0"
    assert abs(scaled.std() - 1.0) < 0.1, "Scaled std not near 1"


# ── Model tests ───────────────────────────────────────────────────────────────

def test_model_trains_and_predicts(sample_df):
    """Model must train without errors and return binary predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    cat_cols = ['Month', 'VisitorType', 'Browser', 'Region', 'TrafficType', 'OperatingSystems']
    X = sample_df.drop('Revenue', axis=1)
    y = sample_df['Revenue'].astype(int)
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_enc['Weekend'] = X_enc['Weekend'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)

    assert set(preds).issubset({0, 1}), "Predictions must be binary"
    assert len(preds) == len(X_test), "Prediction count mismatch"


def test_model_accuracy_above_threshold(sample_df):
    """Model accuracy on synthetic data must exceed 50% (better than random)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    cat_cols = ['Month', 'VisitorType', 'Browser', 'Region', 'TrafficType', 'OperatingSystems']
    X = sample_df.drop('Revenue', axis=1)
    y = sample_df['Revenue'].astype(int)
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_enc['Weekend'] = X_enc['Weekend'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    assert acc > 0.50, f"Accuracy {acc:.2f} is below 50% — worse than random"


def test_model_has_coef_for_all_features(sample_df):
    """Model coefficients count must match number of input features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cat_cols = ['Month', 'VisitorType', 'Browser', 'Region', 'TrafficType', 'OperatingSystems']
    X = sample_df.drop('Revenue', axis=1)
    y = sample_df['Revenue'].astype(int)
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_enc['Weekend'] = X_enc['Weekend'].astype(int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_enc)

    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, random_state=42)
    model.fit(X_sc, y)
    assert model.coef_.shape[1] == X_enc.shape[1], \
        "Coefficient count does not match feature count"


# ── Pipeline file tests ───────────────────────────────────────────────────────

def test_pipeline_file_exists():
    """ubps_pipeline.py must exist."""
    path = os.path.join(os.path.dirname(__file__), '..', 'ubps_pipeline.py')
    assert os.path.exists(path), "ubps_pipeline.py not found"


def test_notebook_file_exists():
    """ubps_notebook.ipynb must exist."""
    path = os.path.join(os.path.dirname(__file__), '..', 'ubps_notebook.ipynb')
    assert os.path.exists(path), "ubps_notebook.ipynb not found"


def test_requirements_file_exists():
    """requirements.txt must exist."""
    path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    assert os.path.exists(path), "requirements.txt not found"
