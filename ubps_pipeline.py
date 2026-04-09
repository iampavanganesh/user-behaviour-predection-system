"""
User Behaviour Prediction System Using Machine Learning
=======================================================
B.Tech Final Year Project

Dataset   : UCI Online Shoppers Purchasing Intention
Algorithm : Logistic Regression (L2, L-BFGS, C=1.0)

Run with: python ubps_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    print('[1/7] Loading dataset...')
    try:
        df = pd.read_csv(URL)
        print(f'      Downloaded from UCI: {df.shape[0]:,} rows x {df.shape[1]} cols')
    except Exception:
        df = pd.read_csv('online_shoppers_intention.csv')
        print(f'      Loaded from local file: {df.shape[0]:,} rows x {df.shape[1]} cols')

    counts = df['Revenue'].value_counts()
    print(f'      No Purchase: {counts[False]:,} ({counts[False]/len(df)*100:.1f}%)')
    print(f'      Purchase   : {counts[True]:,} ({counts[True]/len(df)*100:.1f}%)')
    return df


# =============================================================================
# 2. EDA PLOTS
# =============================================================================
def run_eda(df):
    print('[2/7] Running EDA and saving figures...')
    os.makedirs('figures', exist_ok=True)

    # Fig 1: Class distribution
    counts = df['Revenue'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#5B8DB8', '#E07B54']
    axes[0].bar(['No Purchase', 'Purchase'], [counts[False], counts[True]],
                color=colors, edgecolor='white')
    axes[0].set_title('Target Class Distribution', fontweight='bold')
    axes[0].set_ylabel('Sessions')
    for i, v in enumerate([counts[False], counts[True]]):
        axes[0].text(i, v + 100, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center')
    axes[0].set_ylim(0, 12500)
    axes[1].pie([counts[False], counts[True]],
                labels=['No Purchase (84.5%)', 'Purchase (15.5%)'],
                colors=colors, autopct='%1.1f%%', startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Purchase Split', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig1_class_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Fig 2: PageValues by purchase outcome
    pv_buy   = df[df['Revenue'] == True]['PageValues']
    pv_nobuy = df[df['Revenue'] == False]['PageValues']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pv_nobuy[pv_nobuy < 100], bins=40, alpha=0.7,
            color='#5B8DB8', label=f'No Purchase (mean={pv_nobuy.mean():.2f})')
    ax.hist(pv_buy[pv_buy < 100], bins=40, alpha=0.7,
            color='#E07B54', label=f'Purchase (mean={pv_buy.mean():.2f})')
    ax.set_xlabel('Page Value')
    ax.set_ylabel('Sessions')
    ax.set_title('PageValues by Purchase Outcome', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig2_pagevalues.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Fig 3: Monthly purchase rate
    month_order = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = df.groupby('Month')['Revenue'].agg(['sum','count'])
    monthly['rate'] = monthly['sum'] / monthly['count'] * 100
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(monthly.index, monthly['rate'],
                  color=['#E07B54' if m == 'Nov' else '#5B8DB8' for m in monthly.index],
                  edgecolor='white')
    ax.set_xlabel('Month')
    ax.set_ylabel('Purchase Rate (%)')
    ax.set_title('Purchase Rate by Month (November peak)', fontweight='bold')
    for bar, val in zip(bars, monthly['rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig3_monthly_rate.png', bbox_inches='tight', dpi=150)
    plt.close()

    print('      Figures saved to figures/ folder')


# =============================================================================
# 3. PREPROCESSING PIPELINE
# =============================================================================
def preprocess(df):
    print('[3/7] Preprocessing...')

    # Step 1: Drop nulls
    df_clean = df.dropna().copy()
    print(f'      Dropped {len(df) - len(df_clean)} null rows → {len(df_clean):,} remaining')

    # Step 2: Separate features and target
    X = df_clean.drop('Revenue', axis=1)
    y = df_clean['Revenue'].astype(int)

    # Step 3: One-hot encode categoricals
    cat_cols = ['Month', 'VisitorType', 'Browser', 'Region', 'TrafficType', 'OperatingSystems']
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_enc['Weekend'] = X_enc['Weekend'].astype(int)
    print(f'      Features after encoding: {X_enc.shape[1]}')

    # Step 4: Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'      Train: {len(X_train):,}  |  Test: {len(X_test):,}')

    # Step 5: StandardScaler — fit on train only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Step 6: SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
    print(f'      After SMOTE: {len(X_train_sm):,} training samples (balanced)')

    return X_train_sm, X_test_sc, y_train_sm, y_test, scaler, X_enc.columns.tolist()


# =============================================================================
# 4. TRAIN MODEL
# =============================================================================
def train_model(X_train, y_train):
    print('[4/7] Training Logistic Regression...')
    model = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(f'      Converged in {model.n_iter_[0]} iterations')

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f'      5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    return model


# =============================================================================
# 5. EVALUATE
# =============================================================================
def evaluate(model, X_test, y_test, feature_names):
    print('[5/7] Evaluating model...')

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_pred_prob)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print()
    print('╔══════════════════════════════════════════════════╗')
    print('║      RESULTS — USER BEHAVIOUR PREDICTION         ║')
    print('╠══════════════════════════════════════════════════╣')
    print(f'║  Accuracy   : {acc:.4f}                             ║')
    print(f'║  ROC-AUC    : {auc:.4f}                             ║')
    print(f'║  Macro F1   : {f1_macro:.4f}                             ║')
    print('╚══════════════════════════════════════════════════╝')
    print()
    print(classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=['No Purchase', 'Purchase']).plot(
                               ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig6_confusion_matrix.png', bbox_inches='tight', dpi=150)
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#E07B54', lw=2.5, label=f'LR (AUC={auc:.3f})')
    ax.plot([0,1],[0,1], 'gray', linestyle='--', lw=1.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#E07B54')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig7_roc_curve.png', bbox_inches='tight', dpi=150)
    plt.close()

    return y_pred, y_pred_prob


# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
def feature_importance(model, feature_names):
    print('[6/7] Computing feature importances...')

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False).reset_index(drop=True)

    print('\n  Top 10 Features by Importance:')
    print(f'  {"Rank":<5} {"Feature":<35} {"Coefficient":>12}')
    print('  ' + '-' * 54)
    for i, row in coef_df.head(10).iterrows():
        direction = '+' if row['Coefficient'] > 0 else ''
        print(f'  {i+1:<5} {row["Feature"]:<35} {direction}{row["Coefficient"]:>10.3f}')

    # Coefficient plot
    top_df = coef_df.head(20).sort_values('Coefficient')
    colors = ['#E07B54' if c > 0 else '#5B8DB8' for c in top_df['Coefficient']]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_df['Feature'], top_df['Coefficient'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Coefficient')
    ax.set_title('Top 20 Feature Coefficients', fontweight='bold')
    pos_p = mpatches.Patch(color='#E07B54', label='Increases purchase probability')
    neg_p = mpatches.Patch(color='#5B8DB8', label='Decreases purchase probability')
    ax.legend(handles=[pos_p, neg_p])
    plt.tight_layout()
    plt.savefig('figures/fig8_feature_coefficients.png', bbox_inches='tight', dpi=150)
    plt.close()


# =============================================================================
# 7. SAVE MODEL
# =============================================================================
def save_model(model, scaler):
    print('[7/7] Saving model and scaler...')
    joblib.dump(model,  'ubps_logistic_regression_model.pkl')
    joblib.dump(scaler, 'ubps_scaler.pkl')
    print('      Saved: ubps_logistic_regression_model.pkl')
    print('      Saved: ubps_scaler.pkl')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  USER BEHAVIOUR PREDICTION SYSTEM')
    print('  B.Tech Final Year Project')
    print('=' * 55)
    print()

    df                                     = load_data()
    run_eda(df)
    X_train, X_test, y_train, y_test, \
        scaler, feature_names              = preprocess(df)
    model                                  = train_model(X_train, y_train)
    y_pred, y_pred_prob                    = evaluate(model, X_test, y_test, feature_names)
    feature_importance(model, feature_names)
    save_model(model, scaler)

    print()
    print('All done! Check the figures/ folder for all plots.')
