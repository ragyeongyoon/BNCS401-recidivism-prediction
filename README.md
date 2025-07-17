# Iowa Offender Recidivism Prediction Model
`BNCS401 - Recidivism Prediction using ML`

## 📜 Project Overview
This project develops a machine learning model to predict 3-year recidivism for offenders released from prison in Iowa. It addresses the issue of a low-performing initial model by analyzing the problem from a data imbalance perspective and using class weight adjustments. The process significantly improved the model's recall rate for identifying at-risk individuals, increasing it from **26.5% to 63.8%**.

## 💡 Problem & Solution
### 1. Initial Model Limitations
An initial `Logistic Regression` model showed a recall score of only **26.5%**. This was a critical limitation, as it meant the model was failing to identify nearly three out of every four actual recidivists, making it impractical for real-world use.

| Metric | Initial Model Performance |
| :--- | :--- |
| **Precision** | 0.5607 |
| **Recall** | **0.2645** |
| **F1-Score** | 0.3594 |
| **AUC** | 0.6402 |

### 2. Root Cause Analysis: Data Imbalance
Analysis of the training data revealed a class imbalance, with a ratio of approximately **60:40** for non-recidivists (class 0) to recidivists (class 1). This imbalance caused the model to be biased towards the majority class, resulting in poor performance in detecting the minority class ('recidivist').

### 3. Solution: Class Weight Adjustment
To address this, the `class_weight='balanced'` parameter was applied to the `scikit-learn` model. This technique assigns a higher weight to the minority class during training, forcing the model to pay more attention to its patterns and thereby improving recall.

---
## ⚙️ Development Workflow
### 1. Data Preprocessing
* **Target Variable Creation**: The 'Return to Prison' (Yes/No) column was converted into a numerical `Recidivism` (1/0) column for model training.
* **Feature Selection & Data Leakage Prevention**: To prevent data leakage, features that would only be known *after* recidivism (e.g., 'Days to Return') were excluded. Eight key features known at the time of release were selected.
* **Handling Missing Values**: Rows with missing values in the key features were dropped using `.dropna()`. This refined the dataset from an initial 26,020 records to a final count of **16,438** for the analysis.

### 2. Data Splitting (Train/Test Split)
* The preprocessed data was split into **training (80%)** and **testing (20%)** sets.
* The `stratify=y` option was used to ensure that both the training and testing sets maintained the same class distribution as the original data, creating a reliable evaluation environment.

### 3. Modeling & Evaluation
* Two models, `Logistic Regression` and a `Neural Network`, were trained with the `class_weight='balanced'` option and compared.
* **Recall** was set as the key performance indicator (KPI) to prioritize the project's main goal: "do not miss actual recidivists."

---
## 📈 Final Results
### Model Comparison & Final Selection
The `Balanced Softmax (Logistic Regression)` model was chosen as the final model due to its superior performance, achieving a **Recall of 63.8%**.

| Metric | **Balanced Softmax (Final Model)** | Balanced Neural Network |
| :--- | :--- | :--- |
| **Precision** | 0.4902 | 0.5116 |
| **Recall** | **0.6383** | 0.5818 |
| **F1-Score** | 0.5545 | 0.5444 |

### Confusion Matrix Comparison
*The confusion matrices for the final `Balanced Softmax` model and the `Neural Network` model are shown below. Visually, the `Balanced Softmax` model performs better by making fewer False Negative (FN) errors (480) compared to the `Neural Network` (565), which directly corresponds to its higher recall score.*

---
## 📚 Conclusion & Reflection
* Successfully addressed the impact of data imbalance on model performance, significantly improving **Recall from 26.5% to 63.8%** by applying the `class_weight` technique.
* Learned the importance of selecting the right evaluation metric (e.g., Recall over Accuracy) based on the specific goals of the project.
* Gained practical experience with the **Precision-Recall trade-off**, understanding that improving recall can sometimes lead to a decrease in precision, which requires careful model tuning.

## 💻 Tech Stack
* **Language**: `Python`
* **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `TensorFlow(Keras)`, `Matplotlib`, `Seaborn`
