# Diabetes Prediction Project

This project focuses on predicting the onset of diabetes in Pima Indian women using the **Pima Indians Diabetes Database**. The project began with a basic **Logistic Regression** model and progressively improved over multiple iterations to introduce more advanced models like **XGBoost** and **SHAP** for model explainability.

---

## Dataset Description

The dataset contains medical data for Pima Indian women, including several features known to be important for predicting diabetes:

- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age (years).
- **Outcome**: Class variable (0 for no diabetes, 1 for diabetes).

---

## Version History

### **Version 1: Initial Logistic Regression Model**

The first version of the project was designed to fulfill the initial assignment requirements. In this version, the primary goal was to:
- Implement a basic **Logistic Regression** model to predict the onset of diabetes.
- Perform **Exploratory Data Analysis (EDA)**.
- Recode the `Outcome` variable as a factor.
- Evaluate the model performance using **ROC curve**, **AUC**, and basic accuracy metrics.

### **Version 2: Enhanced EDA and Data Visualization**

This version introduced a deeper **exploratory data analysis (EDA)** to gain more insights into the data:
- **Data Visualizations**: Using `ggplot2` for visualizing distributions and relationships between features.
- **Feature Correlations**: Correlation matrix to explore feature relationships.
- **Data Cleaning**: Missing values and outliers were handled.

While still using the **Logistic Regression** model, this version aimed to better understand the dataset and improve model preparation.

### **Version 3: Logistic Regression Model Tuning and Feature Selection**

In Version 3, we began refining the model:
- **Hyperparameter Tuning**: Feature selection was performed using **AIC** to optimize the logistic regression model.
- **Performance Metrics**: Detailed model evaluation using precision, recall, F1 score, and accuracy.
- **Model Evaluation**: Using metrics like **AUC**, **precision**, and **recall** for performance improvement.

### **Version 4: Advanced Logistic Regression with Interpretability**

In Version 4, we emphasized model **interpretability**:
- **Coefficient Analysis**: Detailed interpretation of logistic regression coefficients to explain feature importance.
- **Confusion Matrix**: Enhanced model evaluation through confusion matrix analysis.
- **Threshold Adjustment**: Investigating classification threshold changes for optimal accuracy.

### **Version 5: Advanced XGBoost Model with SHAP Explanations**

In the final version, the project was significantly advanced with **XGBoost**:
- **XGBoost**: Implemented a gradient boosting algorithm to capture non-linear feature relationships.
- **Bayesian Optimization**: Hyperparameter tuning via **Bayesian optimization** to improve model performance.
- **SHAP Interpretability**: SHAP values were introduced for feature importance analysis and model transparency.
- **Cross-Validation**: A 5-fold cross-validation ensured the model's robustness and generalizability.


---

> **The following applies mainly to the first version (Version 1).**

---


## Modeling Approach
### Logistic Regression
For this project, we use **Logistic Regression** as the primary machine learning algorithm. Logistic regression is ideal for binary classification problems, where the goal is to predict one of two possible outcomes (e.g., diabetes or no diabetes). The logistic regression model estimates the probability that a given instance falls into a specific class (Outcome = 1 for diabetes or Outcome = 0 for no diabetes).

### Key Steps:
1. **Data Preprocessing**:
   - **Handling Missing Values**: Any missing or null values in the dataset were either removed or filled using mean imputation techniques.
   - **Feature Scaling**: Continuous variables such as `Glucose`, `BloodPressure`, `Insulin`, and `BMI` were normalized using **min-max scaling** to ensure all features were on the same scale.
   - **Train-Test Split**: The dataset was split into training and testing sets (80% training, 20% testing) to evaluate the model's performance.

2. **Model Training**:
   - The logistic regression model was trained using the **training set** to learn the relationship between the input variables and the target variable (Outcome).
   - The model's parameters were optimized using the **maximum likelihood estimation (MLE)** technique, which finds the set of coefficients that maximizes the likelihood of the observed data.

3. **Model Evaluation**:
   - The model's performance was evaluated on the **test set** using key metrics such as:
     - **Accuracy**: Proportion of correctly classified instances.
     - **Precision**: Proportion of positive identifications that are actually correct.
     - **Recall (Sensitivity)**: Proportion of actual positives correctly identified by the model.
     - **F1 Score**: Harmonic mean of precision and recall.
   - **Confusion Matrix**: Used to visualize the true positives, true negatives, false positives, and false negatives, providing a detailed insight into model performance.
   - **ROC Curve and AUC Score**: Evaluated the model's ability to distinguish between classes by plotting the true positive rate (sensitivity) against the false positive rate.


## Model Performance
The logistic regression model was trained using relevant features from the diabetes dataset, including **Glucose**, **BMI**, **Age**, **Pregnancies**, **Insulin**, and **BloodPressure**. The performance of the model was evaluated using accuracy and the ROC AUC score.

- **Accuracy**: The model achieved an accuracy of approximately **78.54%** on the test set, indicating that the logistic regression model is fairly effective in predicting diabetes.
  
- **ROC AUC**: The AUC score of **0.8575** shows the model’s strong capability in distinguishing between patients with and without diabetes, demonstrating its effectiveness in classifying individuals based on their likelihood of having diabetes.

### Code Structure
The core logic of the project is implemented in the **`DiabetesPrediction.R`** file. The R script is structured as follows:

1. **Data Loading**: 
   - The dataset `diabetes.csv` is loaded using the `read.csv()` function.
   
2. **Data Preprocessing**:
   - Missing values are handled, and continuous features are normalized.

3. **Modeling**:
   - The `glm()` function in R is used to fit the logistic regression model:
     ```r
     model <- glm(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + Age + DiabetesPedigreeFunction, 
                  family = binomial(link = "logit"), data = train_data)
     ```

4. **Model Evaluation**:
   - The model's predictions on the test set are evaluated using a variety of performance metrics, such as `accuracy`, `precision`, `recall`, and `F1 score`.
   
5. **Visualization**:
   - ROC curve is plotted to evaluate the model’s classification ability:
     ```r
     roc_curve <- roc(test_data$Outcome, predicted_probabilities)
     plot(roc_curve)
     ```


## Results
- **Accuracy**: The model achieved an accuracy of approximately **78.54%** on the test set, indicating that the logistic regression model is fairly effective in predicting diabetes.
- **ROC AUC**: The AUC score of **0.8575** shows the model’s capability in distinguishing between patients with and without diabetes.


## Conclusion
This project demonstrates the use of **Logistic Regression** for predicting diabetes based on medical data. By analyzing key health-related factors, the model provides insights into the factors contributing to diabetes risk and offers a tool for medical professionals to predict the likelihood of diabetes in patients.


## Future Work
- **Hyperparameter Tuning**: Use techniques such as **grid search** or **cross-validation** to further optimize model performance.
- **Feature Engineering**: Explore additional feature engineering techniques, such as polynomial features or interaction terms, to capture more complex relationships.
- **Other Models**: Experiment with other classification models such as **Random Forest**, **Support Vector Machines**, or **XGBoost** to compare performance and improve predictive power.


## Technologies Used
- **R Programming Language**: For data manipulation, modeling, and visualization.
- **Logistic Regression**: For classification modeling.
- **ggplot2**: For plotting and visualization.
- **pROC**: For ROC curve analysis.


## How to Run
1. Clone the repository:
   ```bash
   git clone git@github.com:StellaZhang-Dev/MachineLearning_LogisticRegressionModel_DiabetesPrediction.git
   ```
   
2. Install required packages in R:
   ```r
   install.packages(c("ggplot2", "pROC", "caret"))
   ```

3. Run the R script:
   ```r
   source("DiabetesPrediction.R")
   ```

---


## License
This project is licensed under the MIT License - see the LICENSE file for details.


