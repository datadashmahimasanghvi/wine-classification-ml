# Wine Classification — Italian Cultivars

Chemical analysis of 178 wines from three cultivars grown in the same region of Italy.
End-to-end machine learning project covering EDA, preprocessing, model comparison, and evaluation.

Dataset source: UCI ML Repository — https://archive.ics.uci.edu/dataset/109/wine

---

## Files in this Repository

| File | Description |
|---|---|
| `wine_classification.ipynb` | Main Jupyter / Colab notebook — full analysis |
| `wine_dataset.csv` | Raw dataset from UCI / Kaggle (178 rows, 14 columns) |
| `dashboard.html` | Interactive HTML dashboard — open in any browser |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Google Colab (recommended — zero setup)

1. Open `wine_classification.ipynb` in this repo
2. Click the "Open in Colab" button at the top of the file
3. Run all cells — the notebook loads data automatically via `sklearn.datasets`

### Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/wine-classification.git
cd wine-classification
pip install -r requirements.txt
jupyter notebook wine_classification.ipynb
```

### Interactive Dashboard

Just open `dashboard.html` in any browser. No install or server needed.

---

## Dataset

178 wine samples from three Italian cultivars, each described by 13 chemical features.

| Property | Value |
|---|---|
| Samples | 178 |
| Features | 13 (all continuous) |
| Target classes | 3 cultivars |
| Missing values | None |

The 13 features: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols,
Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315, Proline.

Class distribution: Cultivar I — 59 samples, Cultivar II — 71 samples, Cultivar III — 48 samples.

---

## What the Notebook Covers

**1. Setup and Data Loading**
Loads the dataset, checks for missing values, and prints class distribution.

**2. Exploratory Data Analysis**
- Violin plots showing how each chemical feature is distributed across the three cultivars
- Correlation heatmap showing relationships between all 13 features
- Pairplot of the top 5 most discriminating features
- PCA 2D projection to visualise class separability

**3. Preprocessing**
StandardScaler applied inside pipelines to avoid data leakage. Stratified train/test split (80/20).

**4. Model Training and Comparison**
Five classifiers trained and evaluated with 5-fold cross-validation:
- K-Nearest Neighbours (k=5)
- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest (200 trees)
- Gradient Boosting (200 estimators)

**5. Best Model Evaluation**
Confusion matrix, classification report, and decision boundary visualisation for Random Forest.

**6. Feature Importance**
Bar chart of Random Forest feature importances — Flavanoids and Proline are the top two.

---

## Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | ~97.9% | ~97.8% |
| SVM (RBF) | ~97.9% | ~97.2% |
| Gradient Boosting | ~97.2% | ~97.2% |
| Logistic Regression | ~97.2% | ~97.2% |
| KNN (k=5) | ~95.8% | ~94.4% |

Random Forest is the best overall model. All models exceed 94% — this is a well-posed,
linearly separable classification problem.

Key findings:
- Flavanoids is the single most important feature (importance ≈ 0.22)
- Proline uniquely identifies Cultivar I (mean ≈ 1115 mg/L vs ~520 for others)
- StandardScaler is critical for KNN and SVM
- PCA shows clear separation of all three classes in 2D

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

Install with: `pip install -r requirements.txt`

---

## References

Forina, M. et al. PARVUS — An Extendable Package for Data Exploration, Classification and Correlation.
Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.

UCI ML Repository: https://archive.ics.uci.edu/dataset/109/wine
Kaggle mirror: https://www.kaggle.com/datasets/load_wine

---

## License

MIT — free to use, modify, and distribute with attribution.
