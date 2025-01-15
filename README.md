# # Credit Card Default Prediction

## Overview
This project focuses on building a predictive model to estimate the likelihood of default for credit card clients. By leveraging machine learning techniques and financial data, the project provides insights into the factors influencing defaults and helps financial institutions make informed decisions to manage credit risks effectively.

---

## Dataset
- **Source:** [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Description:**
  - 30,000 instances with 25 features.
  - Contains demographic, behavioral, and financial details of credit card clients.
  - **Target Variable:** `default payment next month` (binary: 1 = Default, 0 = No Default).

---

## Objectives
1. Develop a machine learning model to predict credit card defaults.
2. Analyze key features influencing default probabilities.
3. Evaluate the performance of different machine learning models using metrics like precision, recall, and ROC-AUC.
4. Provide actionable insights for credit risk management.

---

## Features and Techniques
1. **Data Preprocessing:**
   - Handled missing values and standardized numerical features.
   - Encoded categorical variables.
   - Balanced the dataset using **SMOTE** and undersampling techniques.

2. **Exploratory Data Analysis (EDA):**
   - Identified correlations between features and default risk.
   - Visualized trends in client behavior using histograms, box plots, and scatter plots.

3. **Machine Learning Models:**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting

4. **Evaluation Metrics:**
   - Precision, Recall, F1-Score
   - ROC-AUC Curve
   - Confusion Matrix

---

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Platform:** Google Colab

---

## Key Insights
1. **Important Features:**
   - `LIMIT_BAL`: Credit limit strongly correlates with default risk.
   - `PAY_0 to PAY_6`: Payment history significantly impacts default likelihood.
   - `BILL_AMT1 to BILL_AMT6`: High unpaid bill amounts increase default risk.
2. **Balanced Dataset:** SMOTE improved model performance by addressing the class imbalance.
3. **Best Model:** Random Forest achieved the highest accuracy and robust results across all metrics.

---

## Results
- **ROC-AUC Score:** ~0.85 for the Random Forest model.
- **Precision-Recall Tradeoff:** Logistic Regression showed higher precision, while Random Forest balanced precision and recall.
- **Model Recommendations:** Use Random Forest for robust default predictions.

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository_url>
2. Navigate to the project directory:
   ```bash
   cd CreditCardDefaultPrediction
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Open the `Stats_Learning_Project_Final_2.ipynb` notebook in Google Colab or Jupyter Notebook.
5. Upload the dataset (`default_of_credit_card_clients.xls`) to the notebook environment.
6. Run all cells sequentially to:
   - Preprocess the data.
   - Train and evaluate machine learning models.
   - Generate insights and visualizations.

---

## Future Enhancements
1. **Feature Expansion:**
   - Include additional features such as employment history and income levels to enhance prediction accuracy.
2. **Advanced Algorithms:**
   - Experiment with advanced machine learning algorithms like XGBoost, LightGBM, or Neural Networks for better performance.
3. **Interactive Dashboards:**
   - Develop a user-friendly dashboard to visualize predictions and provide insights dynamically.
4. **Explainable AI:**
   - Incorporate SHAP or LIME for better interpretability and transparency of model predictions.

---

## Contributors
- **Shubhang Yadav Sandaveni** 

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.



