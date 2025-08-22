# YouTube Channel View Prediction: A Machine Learning Project

This repository contains a Jupyter Notebook (`Youtube_Prediction_K_Fold_RandomSearch_GridSearchCV_Logarithmic_Transformation.ipynb`) that demonstrates a comprehensive machine learning pipeline for predicting YouTube channel view counts. The project utilizes publicly available metadata and employs robust techniques to build and evaluate a high-performing predictive model.

---

## 1. Project Overview 📊

The primary goal of this project is to build and rigorously evaluate machine learning models to **predict the total video views of YouTube channels**. We tackled common challenges in real-world data, such as highly skewed data distributions and the risk of overfitting, through a systematic and iterative approach. The aim is to provide actionable insights for creators and marketers to estimate channel performance using publicly available data.

---

## 2. Methodology & Key Techniques Used 🛠️

The project followed a multi-stage machine learning methodology:

### 2.1 Data Collection and Source
* **Dataset:** "Global YouTube Statistics" (CSV format).
* **Entries:** 995 distinct global YouTube channels.
* **Features:** Initially 28 diverse features, including channel metadata (e.g., `subscribers`, `uploads`, `category`), performance indicators (e.g., `video_views_rank`, `subscribers_for_last_30_days`), and country-level socioeconomic indicators.
* **Prediction Target:** `video views` (total video views).

### 2.2 Data Preprocessing and Feature Engineering
* **Handling Missing Values:** Numerical features were imputed with their median, and categorical features with their mode.
* **Feature Creation (`created_date_numeric`):** A continuous numerical feature representing channel age was engineered from `created_year` and `created_month`.
* **Categorical Encoding:** `LabelEncoder` was used to convert categorical features (`category`, `Country`, `channel_type`) into numerical format.
* **Logarithmic Transformation of Target:** A **`np.log1p` (logarithmic)** transformation was applied to the `video views` target variable. This was crucial for normalizing its highly skewed distribution, reducing the impact of extreme outliers, and enabling models to learn more stable patterns. Predictions are inverse-transformed using `np.expm1` for evaluation on the original scale.
* **Feature Scaling:** Numerical input features were scaled using `StandardScaler` to normalize their ranges, ensuring fair contribution to model training.

### 2.3 Feature Selection
* **Importance Analysis:** After initial model training and tuning, a comprehensive feature importance analysis was performed for both XGBoost and Random Forest models.
* **Selection:** Based on this analysis, the feature set was strategically reduced to the **top 9 most impactful features** (as identified by the averaged ensemble importances). This step aimed to simplify the models, reduce complexity, mitigate overfitting, and enhance interpretability without compromising predictive power.

    **The 9 Selected Features are:**
    1.  `subscribers`
    2.  `created_date_numeric` (channel age)
    3.  `video_views_rank`
    4.  `channel_type_rank`
    5.  `uploads`
    6.  `video_views_for_the_last_30_days`
    7.  `subscribers_for_last_30_days`
    8.  `category_encoded`
    9.  `lowest_yearly_earnings`

### 2.4 Model Training and Evaluation Strategy
* **Initial Evaluation (Single Split):** An initial 80/20 train-test split on raw data provided a baseline, but revealed high R-squared scores (e.g., ~0.96-0.97) that raised concerns about overfitting.
* **Robust Evaluation (K-Fold Cross-Validation):** To counter overfitting and obtain reliable performance estimates, **5-fold cross-validation** was implemented throughout the entire modeling process. This method ensures models are evaluated on multiple data partitions, providing stable mean and standard deviation for metrics.
* **Hyperparameter Tuning:**
    * **`GridSearchCV` (for XGBoost):** Systematically searched a predefined grid of hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) using cross-validation to find the optimal configuration.
    * **`RandomizedSearchCV` (for Random Forest):** Efficiently sampled a wide range of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`) using cross-validation, which is more effective for larger search spaces.
* **Ensemble Learning (Averaging Ensemble):** The final predictive system combines the predictions of the best-tuned XGBoost and Random Forest models through **simple averaging**. This strategy leverages the diverse learning patterns of individual models to smooth out errors, reduce variance, and provide a more robust and accurate final prediction.

---

## 3. Key Results and Overfitting Mitigation ✅

The iterative process led to highly effective models:

* **Initial Overfitting Confirmation:** The high R-squared scores from the single 80/20 split were significantly reduced when evaluated with k-fold cross-validation on the log-transformed target (e.g., XGBoost R2 dropped from ~0.96 to ~0.40), confirming the initial overfitting.
* **Impact of Log Transformation & K-Fold CV:** These techniques were critical in stabilizing model learning and providing realistic performance estimates.
* **Optimized Ensemble Performance:** The final averaging ensemble model, trained with the top 9 selected features and evaluated on a held-out test set (with predictions converted back to the original scale using `np.expm1`), achieved **outstanding performance**:
    * **MAE (Original Scale): ~669 million views**
    * **R-squared (Original Scale): ~0.9801**
    * **Top Individual Performer:** The **Tuned Random Forest model** (as an individual component) achieved an even more impressive **MAE of ~272 million views** and an **R-squared of ~0.9937** in the final evaluation. This highlights its exceptional predictive power on the original scale.

### Overfitting Precautions:
The project rigorously addresses overfitting through multiple layers of defense:
* **Logarithmic Transformation:** Normalizing the skewed target variable to prevent models from over-focusing on outliers.
* **K-Fold Cross-Validation:** For robust and unbiased evaluation, and for selecting hyperparameters that generalize well.
* **Hyperparameter Tuning:** Optimizing model parameters to strike a balance between bias and variance, ensuring good generalization.
* **Feature Selection:** Reducing model complexity by focusing on only the most impactful features, thereby reducing noise.
* **Ensemble Learning:** Averaging predictions from diverse models to reduce overall variance and improve robustness.

---

## 4. Feature Importance 🌟

The analysis consistently highlighted the most influential features in predicting YouTube video views across both XGBoost and Random Forest models (and their average):

1.  `subscribers`
2.  `created_date_numeric` (channel age)
3.  `video_views_rank`
4.  `channel_type_rank`
5.  `uploads`
6.  `video_views_for_the_last_30_days`
7.  `subscribers_for_last_30_days`
8.  `category_encoded`
9.  `lowest_yearly_earnings`

These features are crucial for understanding the drivers of YouTube channel performance.

---

## 5. Setup Instructions 🛠️

To run this notebook locally, follow these steps:

### 5.1 Prerequisites
Ensure you have Python (3.8+) and `pip` installed on your system. If not, consider installing [Anaconda](https://www.anaconda.com/download/), which simplifies Python and package management.

### 5.2 Install Required Libraries
Open your terminal or command prompt and run the following command to install all necessary Python libraries:
```bash
pip install pandas scikit-learn xgboost numpy matplotlib seaborn scipy
```

### 5.3 Obtain the Dataset
The project uses the "Global YouTube Statistics" dataset.
* **Download:** Ensure you have the `Global-YouTube-Statistics.csv` file. This file should be placed in the **same directory** as your Jupyter Notebook (`.ipynb`) file.

### 5.4 Download the Notebook
Download the `Youtube_Prediction_K_Fold_RandomSearch_GridSearchCV_Logarithmic_Transformation.ipynb` file from its source (e.g., your Google Colab environment via `File > Download > Download .ipynb`).

---

## 6. How to Run the Notebook ▶️

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
