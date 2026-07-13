# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

## Overview

This project builds and evaluates two supervised machine learning models using the cleaned dataset generated in Part 1.

- Regression Model: Predicts the continuous target **AI_Dependency_Score**.
- Classification Model: Predicts whether AI dependency is **High (1)** or **Low (0)** after binarizing the regression target using its median.

The objective is to preprocess the data correctly, avoid data leakage, evaluate model performance, and compare different regularization strategies.

---

# 1. Dataset

Dataset Used:

```
cleaned_data.csv
```

Dataset Size:

- Rows: **14,841**
- Features before encoding: **25**
- Features after one-hot encoding: **53**

---

# 2. Target Variable Definition

### Regression Target

```
AI_Dependency_Score
```

This continuous variable is used for Linear Regression and Ridge Regression.

### Classification Target

A binary target was created using the median of AI_Dependency_Score.

```python
y_clf = (AI_Dependency_Score > median).astype(int)
```

Class Meaning

- 0 → Low AI Dependency
- 1 → High AI Dependency

---

# 3. Data Preprocessing

The following preprocessing steps were performed.

### Missing Values

Missing values were handled before model training.

### Ordinal Encoding

Columns with natural ordering were ordinal encoded.

### High Cardinality Columns Removed

The following text-heavy columns were removed because they contain many unique values and are unsuitable for one-hot encoding.

- Positive_Effects_of_AI
- Negative_Effects_of_AI

### One-Hot Encoding

Nominal categorical variables were converted using One-Hot Encoding.

One-hot encoding prevents introducing false ordinal relationships that would occur if simple label encoding were applied to unordered categories.

After encoding,

```
Number of Features = 53
```

---

# 4. Train-Test Split

The dataset was divided using

```python
train_test_split(
    test_size=0.20,
    random_state=42
)
```

Training Set

```
11,872 samples
```

Testing Set

```
2,969 samples
```

---

# 5. Feature Scaling

StandardScaler was applied.

The scaler was fitted **only on the training data** and then applied to the testing data.

### Why?

Fitting the scaler on the entire dataset would introduce **data leakage**, because statistics from the test data (mean and standard deviation) would influence training.

Using only the training data ensures a fair evaluation.

---

# 6. Linear Regression

Performance

| Metric | Value |
|---------|-------|
| Mean Squared Error | **11.7416** |
| R² Score | **0.9756** |

The model explains approximately **97.56%** of the variance in AI_Dependency_Score.

---

## Most Important Features

| Feature | Coefficient |
|----------|------------:|
| Year | 12.449 |
| AI_Tool_Proficiency | 5.495 |
| Weekly_AI_Usage_Hours | 2.780 |

### Interpretation

A positive coefficient means that increasing the standardized feature by one unit increases the predicted AI_Dependency_Score.

For example,

- Higher Year values are associated with higher predicted AI dependency.
- Greater AI Tool Proficiency increases AI dependency.
- More weekly AI usage leads to higher predicted dependency.

Similarly, a negative coefficient indicates that increasing the feature decreases the predicted target value.

---

# 7. Ridge Regression

Ridge Regression was trained using

```python
alpha = 1.0
```

Performance

| Metric | Value |
|---------|-------|
| Mean Squared Error | **11.7420** |
| R² Score | **0.9756** |

---

## Linear Regression vs Ridge Regression

| Model | MSE | R² Score |
|------|------:|------:|
| Linear Regression | 11.7416 | 0.9756 |
| Ridge Regression | 11.7420 | 0.9756 |

### Interpretation

Both models perform almost identically.

Ridge Regression slightly shrinks the coefficient values to reduce model complexity and multicollinearity.

The **alpha** parameter controls the strength of regularization.

- Small alpha → behaves like Linear Regression.
- Large alpha → stronger coefficient shrinkage.

In this dataset, Ridge Regression does not significantly improve performance because the original model already generalizes well.

---

# 8. Class Imbalance Analysis

Training Class Distribution

| Class | Count |
|------|------:|
| 0 | 5967 |
| 1 | 5905 |

Percentage

| Class | Percentage |
|------|------:|
| 0 | 50.26% |
| 1 | 49.74% |

The dataset is balanced.

Therefore,

**No balancing technique (SMOTE or class weights) was required.**

---

# 9. Logistic Regression

Performance

| Metric | Value |
|---------|-------|
| Accuracy | **96.23%** |
| Precision | **96.79%** |
| Recall | **95.77%** |
| F1 Score | **96.28%** |
| AUC | **0.9951** |

---

# 10. Confusion Matrix

```
               Predicted

             0        1

Actual 0   1409      48

Actual 1    64      1448
```

The model correctly classified the vast majority of observations.

---

# 11. Precision and Recall

### Precision Formula

```
Precision = TP / (TP + FP)
```

Precision measures how many predicted positive cases are actually positive.

---

### Recall Formula

```
Recall = TP / (TP + FN)
```

Recall measures how many actual positive cases were successfully detected.

### Which metric is more important?

For AI dependency prediction, **Recall** is slightly more important because failing to identify users with genuinely high AI dependency (false negatives) may be more costly than incorrectly identifying a few low-dependency users.

---

# 12. ROC Curve and AUC

A ROC curve was generated using predicted probabilities.

AUC Score

```
0.9951
```

### Interpretation

An AUC of 0.9951 indicates excellent class separation.

The model can almost perfectly distinguish between low and high AI dependency.

---

# 13. Decision Threshold Analysis

Threshold comparison

| Threshold | Precision | Recall | F1 Score |
|----------|-----------:|---------:|---------:|
| 0.30 | 0.9390 | 0.9782 | 0.9582 |
| 0.40 | 0.9531 | 0.9683 | 0.9606 |
| 0.50 | **0.9679** | **0.9577** | **0.9628** |
| 0.60 | 0.9774 | 0.9458 | 0.9613 |
| 0.70 | 0.9867 | 0.9325 | 0.9589 |

### Best Threshold

```
0.50
```

The default threshold of **0.50** produced the highest F1 score.

Increasing the threshold improves precision but lowers recall.

Lowering the threshold increases recall while reducing precision.

---

# 14. Logistic Regression Regularization

A second Logistic Regression model was trained using

```
C = 0.01
```

Comparison

| Model | Precision | Recall | F1 | AUC |
|---------|----------:|--------:|--------:|--------:|
| Logistic Regression (C=1.0) | 0.9679 | 0.9577 | 0.9628 | 0.9951 |
| Logistic Regression (C=0.01) | 0.9594 | 0.9524 | 0.9559 | 0.9937 |

### Interpretation

The baseline model (C = 1.0) performed better.

The parameter **C** controls the inverse strength of regularization.

- Larger C → weaker regularization.
- Smaller C → stronger regularization.

Reducing C caused additional coefficient shrinkage, resulting in slightly lower predictive performance.

---

# 15. Bootstrap Confidence Interval

A total of **500 bootstrap samples** were generated.

Results

Mean AUC Difference

```
0.001491
```

95% Confidence Interval

```
Lower Bound : 0.000834

Upper Bound : 0.002201
```

### Interpretation

The confidence interval does **not include zero**.

Therefore, the better performance of the baseline Logistic Regression (C=1.0) is statistically reliable across different samples.

---

# Conclusion

The preprocessing pipeline successfully prevented data leakage and prepared the dataset for machine learning.

Linear Regression achieved an R² score of **0.9756**, demonstrating excellent predictive performance.

For classification, Logistic Regression achieved an **Accuracy of 96.23%** and an **AUC of 0.9951**, indicating excellent discrimination between low and high AI dependency.

Comparisons with Ridge Regression and stronger Logistic Regression regularization showed that the baseline models already provide optimal performance for this dataset.