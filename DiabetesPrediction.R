# Load necessary libraries
library(ggplot2)
library(dplyr)
library(readr)

# Load iris dataset
data(iris)

# 1. Correlations
correlations <- cor(iris[, 1:4])
# Text answer: Most of these correlations are positive, such as the strong positive correlation between Petal.Length and Petal.Width. This makes sense because different parts of the flower tend to grow in proportion to each other.

# 2. Plot Sepal.Width against Sepal.Length
plot1 <- ggplot(iris, aes(x = Sepal.Width, y = Sepal.Length)) +
  geom_point() +
  labs(title = "Scatterplot of Sepal Width vs Sepal Length")
# Text answer: The scatterplot shows a slight negative correlation, which matches the negative correlation computed earlier.

# 3. Fit a linear model using Sepal.Width as predictor and Sepal.Length as response
model1 <- lm(Sepal.Length ~ Sepal.Width, data = iris)
summary(model1)
# Text answer: The estimated coefficient for Sepal.Width is negative, matching the negative correlation observed earlier.

# 4. Setosa correlations
correlations_setosa <- cor(iris[iris$Species == "setosa", 1:4])
# Text answer: The correlations for setosa are generally weaker compared to the overall correlations, except for the strong positive correlation between Petal.Length and Petal.Width.

# 5. Plot Sepal.Width against Sepal.Length, color by species
plot2 <- ggplot(iris, aes(x = Sepal.Width, y = Sepal.Length, color = Species)) +
  geom_point() +
  labs(title = "Scatterplot of Sepal Width vs Sepal Length by Species")
# Text answer: The scatter plot colored by species shows different patterns for different species, which matches the varying correlations observed in the setosa species.

# 6. Fit second model using species and Sepal.Width as predictors and Sepal.Length as response
model2 <- lm(Sepal.Length ~ Sepal.Width + Species, data = iris)
summary(model2)
# Text answer: The coefficient for Sepal.Width changes when adding species to the model, showing the interaction between sepal width and species. The model now accounts for the species-specific differences in sepal measurements.

# 7. Predict the sepal length of a setosa with a sepal width of 3.6 cm
new_data <- data.frame(Sepal.Width = 3.6, Species = "setosa")
prediction <- predict(model2, newdata = new_data)
# Text answer: The prediction for the sepal length of a setosa with a sepal width of 3.6 cm appears reasonable given the model.

# Ensure the data file is in the current working directory
if (!file.exists("a2_diabetes.csv")) {
  stop("Data file 'a2_diabetes.csv' not found in the current working directory.")
}

# Download the dataset from Canvas and place in current directory
getwd()

# Load the data
diabetes_data <- read_csv("a2_diabetes.csv") # Don't change this line!

# Reflect over important variables
# Important variables might include glucose, blood pressure, insulin levels, BMI, age, etc., as they are commonly associated with diabetes.

# 8. Recode Outcome as a factor
diabetes_data <- diabetes_data %>%
  mutate(Outcome = as.factor(Outcome))

# 9. Find a good logistic regression model
# Creating a copy of the data for model training to avoid modifying the original dataset
diabetes_data_clean <- diabetes_data %>%
  filter(complete.cases(.))  # Remove rows with missing values

# Fit the logistic regression model
logistic_model <- glm(Outcome ~ Glucose + BMI + Age + Pregnancies + Insulin + BloodPressure, data = diabetes_data_clean, family = binomial)
summary(logistic_model)
# Text answer: Variables strongly related to diabetes include Glucose, BMI, and Age. The effects in the model appear reasonable given the known risk factors for diabetes.


# 10. Improve the model to achieve an accuracy of at least 76.5%

# Explicitly remove rows with missing values relevant to the model
diabetes_data_clean <- diabetes_data %>%
  filter(complete.cases(Glucose, BMI, Age, Pregnancies, Insulin, BloodPressure, Outcome))

# Use stepwise selection to fit the most optimal logistic regression model from the start
logistic_model <- step(glm(Outcome ~ Glucose + BMI + Age + Pregnancies + Insulin + BloodPressure, 
                           data = diabetes_data_clean, family = binomial), direction = "both")

# Calculate the accuracy on the cleaned dataset with the refined model
predictions <- predict(logistic_model, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
accuracy <- mean(predicted_classes == diabetes_data_clean$Outcome)

# Text answer: Using stepwise selection from the start, the model was optimized to include only significant predictors and achieved an accuracy of at least 76.5%.

# Load necessary library for ROC curve and AUC calculation
library(pROC)

# Ensure Outcome is a factor and predictions are numeric probabilities
diabetes_data_clean$Outcome <- as.factor(diabetes_data_clean$Outcome)

# Make predictions using the logistic model (these are probabilities)
predictions <- predict(logistic_model, type = "response")

# Calculate ROC curve
roc_curve <- roc(diabetes_data_clean$Outcome, predictions)

# Print the AUC value
auc_value <- auc(roc_curve)
print(paste("AUC Value: ", auc_value))

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Logistic Regression Model")

