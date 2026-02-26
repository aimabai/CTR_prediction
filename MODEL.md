# CTR Prediction Model Documentation

## 1. Data Preprocessing

### 1.1 Removing Non-Informative Features
The columns `C9`, `C10`, `C12`, `C18`, and `C25` contained constant values in both the training and test datasets. Since constant features do not contribute to prediction and only introduce noise, they were removed to improve computational efficiency and model focus.

### 1.2 Time Feature Extraction
User click behavior depends strongly on the time of day. Instead of using raw hour values (0–23), which tree models treat as linear, **Cyclic Encoding** was applied to preserve the circular property of time (e.g., ensuring 23:00 is mathematically close to 00:00).



The transformation used is:
$$hour\_sin = \sin\left(\frac{2\pi \cdot hour}{24}\right)$$
$$hour\_cos = \cos\left(\frac{2\pi \cdot hour}{24}\right)$$

### 1.3 Activity and Popularity Features
Two statistical behavioral features were added to approximate user engagement and item attractiveness:

| Feature | Meaning | Proxy For |
| :--- | :--- | :--- |
| **C1_activity** | Number of occurrences of each user ID | User Engagement |
| **C4_popularity** | Number of occurrences of each item ID | Item Attractiveness |

### 1.4 Handling High-Cardinality Categorical Features
Columns `C1` and `C4` contain an extremely large number of unique IDs. Using them directly causes the model to "memorize" specific IDs, leading to severe overfitting. To address this, **Smoothed Target Encoding** was applied:



$$TE = \frac{(count \cdot mean) + (smoothing \cdot global\_mean)}{count + smoothing}$$

* **Smoothing factor (100):** Prevents rare IDs from having extreme weights.
* **K-Fold Strategy:** Out-of-fold encoding was used to prevent data leakage.
* **Native Handling:** All other categorical variables were handled natively by CatBoost’s internal categorical engine.

### 1.5 Train-Validation Split
The dataset was sorted chronologically and split into a **80% Training set** and a **20% Validation set**. This temporal split simulates a real-world production scenario by predicting future clicks from past behavior and prevents temporal leakage.

---

## 2. Model Selection

CTR prediction is a tabular binary classification task characterized by extreme class imbalance (~0.9% positive), high-cardinality IDs, and sparse signals. Gradient Boosting Decision Trees (GBDT) are the industry standard for this type of structured data.

### Why CatBoost?
CatBoost was selected as the primary estimator due to:
* **Native Categorical Processing:** Superior handling of categorical features through ordered boosting.
* **Robustness:** High performance on high-cardinality data with reduced need for manual one-hot encoding.
* **Stability:** Reliable performance under severe class imbalance without requiring extreme resampling.

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Depth** | 7 |
| **Learning Rate** | 0.03 |
| **Iterations** | 2000 |
| **L2 Regularization** | 10 |
| **Early Stopping** | 150 rounds |

---

## 3. Evaluation Metrics

Due to the ~0.9% click rate, accuracy is not a meaningful metric. The following metrics were used:

* **AUC (Primary Metric):** Measures the ranking quality. It evaluates the model's ability to rank a positive instance higher than a random negative instance.
* **LogLoss (Secondary Metric):** Measures probability calibration. This ensures the predicted probability is a reliable estimate of the actual click likelihood, which is critical for bidding engines.



---

## Appendix — LLM Usage

An LLM was utilized during this assignment for the following purposes:
* Improving code readability, modularity, and structure.
* Writing and formatting technical documentation (`MODEL.md`).
* Clarifying theoretical explanations of preprocessing techniques (e.g., Cyclic Encoding and Target Smoothing).

*Note: All feature engineering decisions, experimental design, and hyperparameter tuning were directed and verified by the student.*