# DiabetesPredictionProject


## Project Overview
This project aims to predict the likelihood of diabetes using a **Logistic Regression Model**. The dataset used for this project contains various medical and demographic variables that are analyzed to determine the factors most strongly associated with diabetes risk. The ultimate goal is to build a predictive model that can classify individuals based on their risk of developing diabetes.


## Dataset
The dataset used in this project is **`diabetes.csv`**, which includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: The target variable, indicating whether the patient has diabetes (1) or not (0)

The dataset is sourced from medical diagnostic records and focuses on predicting the presence of diabetes in women.


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


## License
This project is licensed under the MIT License - see the LICENSE file for details.


