# Customer Churn Prediction: Machine Learning for Customer Retention

## Project Overview
Customer churn (when customers leave a service) is one of the most expensive problems for any business. It costs significantly more to acquire a new customer than to retain an existing one. 

The goal of this project is to build a Machine Learning model that can predict **which customers are at the highest risk of leaving**, allowing the business to proactively intervene with retention strategies (like discounts or targeted marketing).

## Tools & Technologies Used
* **Python** (Data manipulation and modeling)
* **Pandas & NumPy** (Data cleaning and preprocessing)
* **Scikit-Learn** (Machine Learning models, scaling, metrics)
* **Imbalanced-Learn (SMOTE)** (Handling class imbalance)
* **Matplotlib & Seaborn** (Data visualization and feature importance)

## The Machine Learning Process
### 1. Data Preprocessing
* Handled missing values and one-hot encoded categorical variables.
* Scaled numerical features (`Tenure` and `TotalCharges`) using `StandardScaler` to ensure distance-based models evaluate the data evenly.
* Performed an 80/20 Train-Test split to ensure the model was evaluated on unseen data.

### 2. Model Selection & Baseline Testing
Three models were initially built and evaluated:
1. **Logistic Regression** (Baseline & highly interpretable)
2. **K-Nearest Neighbors (KNN)** (Distance-based clustering)
3. **Random Forest Classifier** (Tree-based ensemble)

**Initial Findings:** While Accuracy was high (~78%), the **Recall** for churning customers was dangerously low (around 49%). Because the dataset was highly imbalanced (many more loyal customers than churners), the models were heavily biased toward guessing that customers would stay.

### 3. Handling Class Imbalance with SMOTE
In churn prediction, **Recall** is the most important metric (missing a churner costs the company money). To fix the bias, I applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data. This generated synthetic churn examples, creating a perfectly balanced 50/50 training environment.

**The Results after SMOTE:**
* The Logistic Regression model became the most effective model for the business.
* **Recall skyrocketed from 49% to 66%**. The model is now actively catching significantly more flight-risk customers.

## Key Business Insights (Feature Importance)
By extracting the mathematical coefficients from the Logistic Regression model, I was able to determine exactly what drives customer behavior:

🔴 **Top Driver of Churn: `Phone Service`**
Customers with standard phone service are highly likely to leave. This indicates that phone service alone is not "sticky" enough to keep customers tied to the ecosystem.

🟢 **Top Driver of Loyalty: `Online Security`**
Customers who utilize Online Security are the least likely to leave. Setting up security is a high-effort task, meaning these customers are deeply embedded in the company's services.

**Business Recommendation:** Launch a campaign offering a free 3-month trial of Online Security to all Phone Service customers. Converting them into the security ecosystem will drastically lower their risk of churning.

## How to Run the Code
1. Clone this repository to your local machine.
2. Ensure you have the necessary libraries installed (`pip install pandas scikit-learn imbalanced-learn matplotlib seaborn`).
3. Run the `Model.ipynb` Jupyter Notebook from top to bottom to view the data cleaning, model training, and visual insights.
