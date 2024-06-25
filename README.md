# Predictive Modeling on Heart Disease Data
This R script performs predictive modeling on heart disease data using various machine learning techniques. Below is a breakdown of the key sections and their functionalities:

1.Data Preparation and Initial Analysis

Loads necessary packages (ISLR, dplyr, ggplot2).
Reads the heart disease dataset (HEART2.CSV).
Performs exploratory data analysis (EDA) with str() and barplot() functions.
Preprocesses the target variable (tgt) for logistic regression.
2.Logistic Regression

Fits a logistic regression model (glm) to predict heart disease (tgt) based on age, sex, and cholesterol (chol).
Calculates and interprets logistic regression coefficients (summary(glm1)).
Computes probabilities (P45, P55) of heart disease for specific age, sex, and cholesterol values.
Makes predictions (glm.pred) and evaluates model accuracy using a confusion matrix.
3.Linear Discriminant Analysis (LDA)

Splits the data into training and test sets (train, H.train, H.test).
Uses Linear Discriminant Analysis (lda) to classify heart disease presence (tgt2) based on age, sex, and cholesterol.
Evaluates LDA model performance with predictions (lda.pred) and a confusion matrix.
4.Quadratic Discriminant Analysis (QDA)

Applies Quadratic Discriminant Analysis (qda) to predict heart disease presence.
Assesses QDA model accuracy using predictions and a confusion matrix.
5.K-Nearest Neighbors (KNN)

Utilizes K-Nearest Neighbors (knn) for classification.
Computes predictions (knn.pred) based on nearest neighbors and evaluates accuracy with a confusion matrix.
6.Cross-Validation and Error Metrics

Implements cross-validation using caret package (createDataPartition).
Defines an error metric function (err_metric) to calculate precision, recall, F1 score, and accuracy from confusion matrices.
7.Logistic Regression (Advanced)

Performs logistic regression (logit_m) on training data (train_data).
Predicts heart disease using test data and evaluates predictions with a confusion matrix and error metrics.
Plots an ROC curve (roc_score) to assess model performance.
