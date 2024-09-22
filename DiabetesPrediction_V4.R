######################### Section 1: Library and Data Loading #########################

# Loading necessary libraries for data manipulation, visualization, and model evaluation
library(ggplot2)
library(dplyr)
library(readr)
library(caret)  # For model training, cross-validation, and hyperparameter tuning
library(pROC)  # For ROC and AUC calculations
library(randomForest)  # Random Forest for comparison
library(e1071)  # For SVM model comparison
library(DMwR)  # For handling imbalanced data (SMOTE)
library(xgboost)  # For XGBoost model
library(LIME)  # For model interpretation
library(tidyverse)
library(mlr)  # For model stacking and more advanced techniques
library(SHAPforxgboost)  # For SHAP explanations

# Loading the dataset
diabetes_data <- read_csv("a2_diabetes.csv")

######################### Section 2: Exploratory Data Analysis (EDA) #########################

# Summarize the structure and contents of the dataset
summary(diabetes_data)
str(diabetes_data)

# Checking for missing values
missing_data <- colSums(is.na(diabetes_data))
print("Missing data per column:")
print(missing_data)

# Visualize the distribution of the outcome variable (diabetes outcome)
ggplot(diabetes_data, aes(x = factor(Outcome))) + 
  geom_bar(fill = "steelblue") + 
  labs(title = "Distribution of Diabetes Outcome", x = "Outcome", y = "Count")

# Correlation matrix for numeric variables
numeric_columns <- diabetes_data %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_columns, use = "complete.obs")
print(correlation_matrix)

######################### Section 3: Data Cleaning and Preprocessing #########################

# Handle missing data by removing rows with missing values (for simplicity)
diabetes_data_clean <- diabetes_data %>% 
  filter(complete.cases(.))  # Remove rows with missing values

# Confirm the dimensions of the cleaned dataset
print(paste("Data after cleaning, number of rows:", nrow(diabetes_data_clean)))

######################### Section 4: Feature Scaling #########################

# Scaling numeric features to ensure comparability
diabetes_data_clean_scaled <- diabetes_data_clean
scaling_columns <- c("Glucose", "BMI", "Age", "Pregnancies", "Insulin", "BloodPressure")
diabetes_data_clean_scaled[scaling_columns] <- scale(diabetes_data_clean_scaled[scaling_columns])

######################### Section 5: Hyperparameter Tuning with Grid Search #########################

# Setting up a grid search for SVM hyperparameters
set.seed(123)
svm_tune_grid <- expand.grid(C = seq(0.1, 1, by = 0.1),
                             gamma = seq(0.01, 0.1, by = 0.01))

svm_tune_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

# Fit the SVM model with grid search for optimal hyperparameters
svm_tuned <- train(Outcome ~ Glucose + BMI + Age + Pregnancies + Insulin + BloodPressure,
                   data = diabetes_data_clean_scaled,
                   method = "svmRadial",
                   tuneGrid = svm_tune_grid,
                   trControl = svm_tune_control)

print("Tuned SVM Model with Hyperparameters:")
print(svm_tuned)

######################### Section 6: Recursive Feature Elimination (RFE) #########################

# Feature selection using RFE
set.seed(123)
rfe_control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rfe_result <- rfe(diabetes_data_clean_scaled[, -ncol(diabetes_data_clean_scaled)], diabetes_data_clean_scaled$Outcome,
                  sizes = c(1:5), rfeControl = rfe_control)

# Print selected features
print("Selected features from RFE:")
print(predictors(rfe_result))

######################### Section 7: XGBoost Model #########################

# Training an XGBoost model
xgb_train_matrix <- xgb.DMatrix(data = as.matrix(diabetes_data_clean_scaled[, scaling_columns]), 
                                label = diabetes_data_clean_scaled$Outcome)

xgb_params <- list(objective = "binary:logistic", eval_metric = "auc")

set.seed(123)
xgb_model <- xgb.train(params = xgb_params, 
                       data = xgb_train_matrix, 
                       nrounds = 100, 
                       watchlist = list(train = xgb_train_matrix), 
                       verbose = 0)

print("XGBoost Model:")
print(xgb_model)

######################### Section 8: Model Stacking #########################

# Create base learners for stacking
base_learners <- list(
  logistic = makeLearner("classif.logreg"),
  randomForest = makeLearner("classif.randomForest", par.vals = list(ntree = 100)),
  svm = makeLearner("classif.svm", kernel = "radial")
)

# Create the meta-learner (logistic regression for stacking)
stack_learner <- makeStackedLearner(base.learners = base_learners, 
                                    predict.type = "prob", 
                                    method = "stack.cv")

# Train the stacked model
task <- makeClassifTask(data = diabetes_data_clean_scaled, target = "Outcome")
set.seed(123)
stacked_model <- train(stack_learner, task)

# Print the stacked model
print("Stacked Model:")
print(stacked_model)

######################### Section 9: SHAP Explanations for XGBoost #########################

# Compute SHAP values for XGBoost
shap_values <- shap.values(xgb_model, X_train = as.matrix(diabetes_data_clean_scaled[, scaling_columns]))

# Plot SHAP summary
shap.plot.summary(shap_values)
print("SHAP values computed and plotted.")

######################### Section 10: Conclusion #########################

# The models have been trained, evaluated, and explained.
# XGBoost, SVM with hyperparameter tuning, and stacked models have been explored.
# SHAP values provide model explainability, especially for the XGBoost model.

