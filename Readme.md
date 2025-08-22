# YouTube Channel View Prediction: A Machine Learning Project

This repository contains a Jupyter Notebook (`Youtube_Prediction_K_Fold_RandomSearch_GridSearchCV_Logarithmic_Transformation.ipynb`) that demonstrates a comprehensive machine learning pipeline for predicting YouTube channel view counts. The project utilizes publicly available metadata and employs robust techniques to build and evaluate a high-performing predictive model.

---

## Project Overview 📊

The goal of this project is to predict the total video views of YouTube channels using machine learning. We address challenges inherent in real-world data, such as skewed distributions and the risk of overfitting, through a systematic and iterative approach.

### Key Techniques Used:
* **Data Preprocessing:** Handling missing values, feature engineering (`created_date_numeric`), and categorical encoding.
* **Logarithmic Transformation:** Applied to the target variable (`video views`) to manage its highly skewed distribution and improve model learning.
* **K-Fold Cross-Validation:** For robust and unbiased model evaluation, and to detect overfitting.
* **Hyperparameter Tuning:** Systematic optimization of model parameters using `GridSearchCV` (for XGBoost) and `RandomizedSearchCV` (for Random Forest).
* **Ensemble Learning:** Combining the predictions of the best-tuned XGBoost and Random Forest models through simple averaging for enhanced accuracy and robustness.
* **Feature Selection:** Identifying and utilizing the most impactful features to streamline the model and improve interpretability.

---

## Features

The models are trained on 10 selected features identified as most impactful:
* `subscribers`
* `video_views_rank`
* `created_date_numeric`
* `channel_type_rank`
* `uploads`
* `video_views_for_the_last_30_days`
* `subscribers_for_last_30_days`
* `category_encoded`
* `highest_yearly_earnings`
* `lowest_yearly_earnings`

### Prediction Target:
* `video views` (total video views)

---

## Setup Instructions 🛠️

To run this notebook locally, follow these steps:

### 1. Prerequisites
Ensure you have Python (3.8+) and `pip` installed on your system. If not, consider installing [Anaconda](https://www.anaconda.com/download/), which simplifies Python and package management.

### 2. Install Required Libraries
Open your terminal or command prompt and run the following command to install all necessary Python libraries:
```bash
pip install pandas scikit-learn xgboost numpy matplotlib seaborn scipy
```

### 3. Obtain the Dataset
The project uses the "Global YouTube Statistics" dataset.
* **Download:** Ensure you have the `Global-YouTube-Statistics.csv` file. This file should be placed in the **same directory** as your Jupyter Notebook (`.ipynb`) file.

### 4. Download the Notebook
Download the `Youtube_Prediction_K_Fold_RandomSearch_GridSearchCV_Logarithmic_Transformation.ipynb` file from its source (e.g., your Google Colab environment via `File > Download > Download .ipynb`).

---

## How to Run the Notebook ▶️

1.  **Navigate to Project Directory:** Open your terminal or command prompt and navigate to the directory where you saved the `.ipynb` file and the `Global-YouTube-Statistics.csv` dataset.
    ```bash
    cd path/to/your/project
    ```
2.  **Launch Jupyter Notebook:** Run the following command:
    ```bash
    jupyter notebook
    ```
    This will open a new tab in your web browser, displaying the Jupyter Notebook dashboard.
3.  **Open the Notebook:** Click on `Youtube_Prediction_K_Fold_RandomSearch_GridSearchCV_Logarithmic_Transformation.ipynb` to open it.
4.  **Execute Cells:** Run all cells sequentially from top to bottom. You can do this by selecting `Cell > Run All` from the menu, or by clicking each cell and pressing `Shift + Enter`.

---

## Key Results and Overfitting Mitigation ✅

After running the notebook, you will observe the following:

* **Initial Overfitting:** An initial evaluation with a simple 80/20 split on untransformed data will show deceptively high R-squared scores (e.g., ~0.96-0.97), indicating overfitting.
* **Impact of Log Transformation & K-Fold CV:** Subsequent k-fold cross-validation with a logarithmic transformation of the target variable will reveal more realistic R-squared scores on the log-scale (e.g., ~0.3-0.4), confirming the initial overfitting and demonstrating improved model stability.
* **Optimized Ensemble Performance:** The final averaging ensemble model, trained with the top 10 selected features, achieves outstanding performance on the **original scale of video views**:
    * **MAE (Original Scale): ~695 million views**
    * **R-squared (Original Scale): ~0.9757**
    (Note: The Tuned XGBoost model, as an individual component, achieved an even higher R-squared of ~0.9834 and a lower MAE of ~654 million views in the final evaluation.)

### Overfitting Precautions:
The project rigorously addresses overfitting through:
* **Logarithmic Transformation:** Normalizing the skewed target variable.
* **K-Fold Cross-Validation:** For robust evaluation and parameter selection.
* **Hyperparameter Tuning:** Optimizing model parameters to generalize well.
* **Feature Selection:** Reducing model complexity by focusing on the most impactful features.
* **Ensemble Learning:** Averaging predictions from diverse models to reduce variance.

---

## Feature Importance 🌟

The analysis highlights the most influential features in predicting YouTube video views:
1.  `subscribers`
2.  `created_date_numeric` (channel age)
3.  `video_views_rank`
4.  `uploads`
5.  `channel_type_rank`
6.  `category_encoded`
7.  `video_views_for_the_last_30_days`
8.  `subscribers_for_last_30_days`
9.  `lowest_yearly_earnings`

---

## Contact 📧

For any questions or further collaboration, please contact [Your Name/Email].

"""