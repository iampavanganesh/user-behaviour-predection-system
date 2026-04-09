# User Behaviour Prediction System Using Machine Learning

**B.Tech Final Year Project**

Predicts whether an online shopper will complete a purchase based on their browsing session behaviour — pages visited, time spent, bounce rates, visitor type, and more — using Logistic Regression with SMOTE oversampling on the UCI Online Shoppers dataset.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | **0.86** |
| ROC-AUC | **0.91** |
| Macro F1 | **0.82** |
| Class 1 Recall | **0.78** |

---

## Project Structure

```
UserBehaviourPredictionSystem/
├── ubps_notebook.ipynb          ← Main Jupyter notebook (run this)
├── ubps_pipeline.py             ← Clean Python script version
├── requirements.txt
├── figures/                     ← Auto-generated plots (created on run)
│   ├── fig1_class_distribution.png
│   ├── fig2_pagevalues.png
│   ├── fig3_monthly_rate.png
│   ├── fig4_visitor_type.png
│   ├── fig5_correlation_heatmap.png
│   ├── fig6_confusion_matrix.png
│   ├── fig7_roc_curve.png
│   └── fig8_feature_coefficients.png
├── tests/
│   └── test_ubps.py
├── LICENSE
└── README.md
```

> **Generated after running:** `ubps_logistic_regression_model.pkl`, `ubps_scaler.pkl`, `online_shoppers_intention.csv`

---

## Dataset

**UCI Online Shoppers Purchasing Intention Dataset**
- 12,330 session records
- 18 features (10 numerical, 8 categorical)
- Target: `Revenue` — whether the session ended in a purchase
- Class imbalance: 84.5% No Purchase / 15.5% Purchase

Dataset downloads automatically from UCI on first run. No manual download needed.

UCI Repository: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

---

## How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/UserBehaviourPredictionSystem.git
cd UserBehaviourPredictionSystem
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4a — Run as Jupyter Notebook (recommended)

```bash
jupyter notebook
```

Open `ubps_notebook.ipynb` → **Kernel → Restart & Run All**

### Step 4b — Run as Python script

```bash
python ubps_pipeline.py
```

### Step 5 — Run tests

```bash
pytest tests/
```

---

## ML Pipeline

```
Raw Data (12,330 records)
      ↓
Drop 14 null rows → 12,316 records
      ↓
One-hot encode: Month, VisitorType, Browser, Region, TrafficType, OS
      ↓
Stratified 80/20 Train-Test Split
      ↓
StandardScaler (fit on train only)
      ↓
SMOTE oversampling on training set only
      ↓
LogisticRegression (L2, C=1.0, lbfgs, max_iter=1000)
      ↓
Evaluate on held-out test set
```

---

## Key Findings

- **PageValues** is the strongest predictor of purchase (+2.84 coefficient)
- **November** has the highest purchase rate (~25%) due to Black Friday
- **Returning Visitors** convert at higher rates than new visitors
- **Exit Rate** and **Bounce Rate** are strong negative predictors
- SMOTE effectively addresses the 84.5% / 15.5% class imbalance

---

## Technologies Used

| Library | Purpose |
|---|---|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Logistic Regression, scaling, metrics, cross-validation |
| imbalanced-learn | SMOTE oversampling |
| matplotlib, seaborn | Visualisation (8 figures) |
| joblib | Model serialisation |
| jupyter | Interactive notebook environment |

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Author

**Ganesh** — B.Tech CSE Final Year
