######################### Section 1: Library and Data Loading #########################

# Load the necessary libraries
library(xgboost)
library(caret)  # For model training, cross-validation, and hyperparameter tuning
library(mlr)    # For hyperparameter tuning
library(SHAPforxgboost)  # For SHAP explanations
library(ggplot2)
library(dplyr)
library(readr)
library(pROC)  # For ROC and AUC calculations

# Load the dataset
diabetes_data <- read_csv("a2_diabetes.csv")

######################### Section 2: Data Preprocessing #########################

# Handle missing data by removing rows with missing values (for simplicity)
diabetes_data_clean <- diabetes_data %>% filter(complete.cases(.))

# Scale numeric features to ensure comparability
scaling_columns <- c("Glucose", "BMI", "Age", "Pregnancies", "Insulin", "BloodPressure")
diabetes_data_clean[scaling_columns] <- scale(diabetes_data_clean[scaling_columns])

######################### Section 3: Hyperparameter Tuning for XGBoost #########################

# Set up cross-validation and hyperparameter tuning with Bayesian Optimization
set.seed(123)

# Define the parameter space for XGBoost
xgb_param_space <- makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.3),    # Learning rate
  makeNumericParam("max_depth", lower = 3, upper = 10),   # Max tree depth
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),  # Feature sampling
  makeNumericParam("subsample", lower = 0.5, upper = 1),  # Row sampling
  makeNumericParam("min_child_weight", lower = 1, upper = 10),   # Minimum sum of instance weight (hessian) needed in a child
  makeNumericParam("lambda", lower = 0, upper = 10),  # L2 regularization term
  makeNumericParam("alpha", lower = 0, upper = 10)    # L1 regularization term
)

# Define the XGBoost learner
xgb_learner <- makeLearner("classif.xgboost", predict.type = "prob", eval_metric = "auc")

# Set up cross-validation and tuning control
xgb_tune_control <- makeTuneControlBayes(budget = 50)

# Perform 5-fold cross-validation
xgb_cv <- makeResampleDesc("CV", iters = 5)

# Define the task
task <- makeClassifTask(data = diabetes_data_clean, target = "Outcome")

# Tune the model using Bayesian optimization
xgb_tuned <- tuneParams(xgb_learner, task = task, resampling = xgb_cv, par.set = xgb_param_space, control = xgb_tune_control, measures = auc)

# Print the best hyperparameters
print(xgb_tuned$x)

# Train the final XGBoost model using the tuned hyperparameters
final_xgb_model <- setHyperPars(xgb_learner, par.vals = xgb_tuned$x)
xgb_trained_model <- train(final_xgb_model, task)

######################### Section 4: Model Evaluation #########################

# Predict on the training data to evaluate performance
xgb_predictions <- predict(xgb_trained_model, task)

# Calculate the AUC score
xgb_auc <- performance(xgb_predictions, measures = auc)
print(paste("XGBoost AUC: ", xgb_auc))

######################### Section 5: SHAP Model Explanations #########################

# Convert the data to a DMatrix for SHAP
xgb_train_matrix <- xgb.DMatrix(data = as.matrix(diabetes_data_clean[, scaling_columns]), label = diabetes_data_clean$Outcome)

# Calculate SHAP values
shap_values <- shap.values(xgb_trained_model$learner.model, X_train = as.matrix(diabetes_data_clean[, scaling_columns]))

# Plot SHAP summary
shap.plot.summary(shap_values)

# SHAP dependence plot for the most important features
shap.plot.dependence(shap_values, feature = "Glucose", X_train = as.matrix(diabetes_data_clean[, scaling_columns]))

######################### Section 6: Conclusion #########################

# The model has been fine-tuned using Bayesian optimization, evaluated with cross-validation, and explained using SHAP values.
# The AUC score and SHAP plots illustrate the model's performance and the contribution of key features.

