# Install and load necessary packages if not already installed
if (!require(ISLR)) install.packages("ISLR")
if (!require(dplyr)) install.packages("dplyr")
if (!require(MASS)) install.packages("MASS")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")

library(ISLR)
library(dplyr)
library(MASS)
library(caret)
library(pROC)

# Load the dataset
Hdat2 <- read.table("HEART2.CSV", sep = ",", header = TRUE)

# Explore the structure of the dataset
str(Hdat2)

# Data exploration: bar plots for target and sex variables
barplot(table(Hdat2$target), main = "Distribution of Target Variable")
barplot(table(Hdat2$sex), main = "Distribution of Sex Variable")

# Preprocess the target variable for binary classification
Hdat2$tgt <- ifelse(Hdat2$target == "yes", 0, 1)

# Logistic regression model
glm1 <- glm(tgt ~ age + sex + chol, data = Hdat2, family = binomial)
summary(glm1)

# Calculate probabilities for specific age, sex, and cholesterol values
Age <- 45
male <- 1
chol <- 202
L <- -5.954 + 0.063 * Age + 1.676 * male + 0.0047 * chol
P45 <- exp(L) / (1 + exp(L)) * 100

Age <- 55
L55 <- -5.954 + 0.063 * Age + 1.676 * male + 0.0047 * chol
P55 <- exp(L55) / (1 + exp(L55)) * 100

# Predict probabilities for the entire dataset
glm.probs <- predict(glm1, type = "response")
head(glm.probs)

# Create predictions based on probability threshold 0.5
glm.pred <- rep("No", nrow(Hdat2))
glm.pred[glm.probs > 0.5] <- "Yes"
table(glm.pred, Hdat2$tgt)

# Accuracy calculation
accuracy <- sum(glm.pred == Hdat2$tgt) / length(Hdat2$tgt) * 100
cat("Accuracy:", accuracy, "%\n\n")

# Data partitioning for training and testing
set.seed(2)
train_indices <- sample(1:nrow(Hdat2), 0.8 * nrow(Hdat2))
H.train <- Hdat2[train_indices, ]
H.test <- Hdat2[-train_indices, ]

# Linear Discriminant Analysis (LDA)
lda.fit <- lda(tgt ~ age + sex + chol, data = H.train)
plot(lda.fit)

lda.pred <- predict(lda.fit, H.test)
lda.class <- lda.pred$class
table(lda.class, H.test$tgt)

# Quadratic Discriminant Analysis (QDA)
qda.fit <- qda(tgt ~ age + sex + chol, data = H.train)
qda.pred <- predict(qda.fit, H.test)$class
table(qda.pred, H.test$tgt)

# Scaling data and preparing for KNN classification
St.X <- scale(Hdat2[, -c(3, 15)])
targetf <- as.factor(Hdat2$target)
test_indices <- 1:100
train.X <- St.X[-test_indices, ]
test.X <- St.X[test_indices, ]
train.Y <- targetf[-test_indices]
test.Y <- targetf[test_indices]

# KNN classification
knn.pred <- knn(train.X, test.X, train.Y, k = 1)
result_table <- table(knn.pred, test.Y)
print(result_table)

# Error metrics function for confusion matrix
err_metric <- function(CM) {
  TN <- CM[1, 1]
  TP <- CM[2, 2]
  FP <- CM[1, 2]
  FN <- CM[2, 1]
  precision <- TP / (TP + FP)
  recall_score <- FP / (FP + TN)
  f1_score <- 2 * ((precision * recall_score) / (precision + recall_score))
  accuracy_model <- (TP + TN) / (TP + TN + FP + FN)
  False_positive_rate <- FP / (FP + TN)
  False_negative_rate <- FN / (FN + TP)
  cat(paste("Precision value of the model:", round(precision, 2), "\n"))
  cat(paste("Accuracy of the model:", round(accuracy_model, 2), "\n"))
  cat(paste("Recall value of the model:", round(recall_score, 2), "\n"))
  cat(paste("False Positive rate of the model:", round(False_positive_rate, 2), "\n"))
  cat(paste("False Negative rate of the model:", round(False_negative_rate, 2), "\n"))
  cat(paste("f1 score of the model:", round(f1_score, 2), "\n"))
}

# Data partitioning using caret package
set.seed(101)
split <- createDataPartition(Hdat2$tgt, p = 0.80, list = FALSE)
train_data <- Hdat2[split, ]
test_data <- Hdat2[-split, ]

# Logistic regression with caret
logit_m <- glm(formula = tgt ~ age + sex + chol, data = train_data, family = 'binomial')
summary(logit_m)

logit_P <- predict(logit_m, newdata = test_data[-15], type = 'response')
logit_P <- ifelse(logit_P > 0.5, 1, 0)
CM <- table(test_data[, 13], logit_P)
print(CM)
err_metric(CM)

# ROC curve using pROC library
roc_score <- roc(test_data[, 15], logit_P)
plot(roc_score, main = "ROC Curve - Logistic Regression")

# End of script


