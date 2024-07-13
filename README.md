# Heart Disease Analysis with R

This repository contains R scripts for analyzing a heart disease dataset (`HEART2.CSV`) using various statistical and machine learning techniques. The dataset includes information such as age, sex, cholesterol levels, and whether the individual has heart disease.

## Dataset Description

The dataset (`HEART2.CSV`) consists of the following columns:

- `age`: Age of the individual
- `sex`: Gender of the individual (1 = male, 0 = female)
- `chol`: Cholesterol level
- `target`: Whether the individual has heart disease ("yes" or "no")

## Scripts and Analyses

### 1. Data Exploration and Preprocessing

- **Data Loading**: The dataset is loaded using `read.table` and explored using `str()` to understand its structure.
- **Exploratory Data Analysis**: Bar plots are created to visualize the distribution of the `target` and `sex` variables.

### 2. Logistic Regression Model

- **Model Training**: A logistic regression model (`glm`) is trained to predict the probability of heart disease based on age, sex, and cholesterol levels.
- **Model Evaluation**: The model's performance is evaluated using summary statistics and predictions are made for specific age and sex scenarios.

### 3. Classification Models

- **Linear Discriminant Analysis (LDA)**: LDA is applied to classify individuals into heart disease categories.
- **Quadratic Discriminant Analysis (QDA)**: QDA is used as an alternative classification method.
- **K-Nearest Neighbors (KNN)**: KNN classification is performed using `knn` from the `caret` package.

### 4. Error Metrics and Evaluation

- **Confusion Matrix**: Error metrics such as precision, recall, false positive rate, false negative rate, and F1 score are calculated using a custom function (`err_metric`).
- **ROC Curve**: The ROC curve is plotted using the `pROC` package to visualize the performance of the logistic regression model.

### 5. Data Partitioning and Validation

- **Data Partitioning**: The dataset is split into training and testing sets using `createDataPartition` from the `caret` package.
- **Validation**: Models are trained on the training set and validated on the testing set to assess their generalization ability.

## Requirements

- R (version 3.5 or higher)
- R packages: `ISLR`, `dplyr`, `MASS`, `caret`, `pROC`

## Usage

1. Ensure R and the required packages are installed.
2. Download or clone this repository.
3. Place your `HEART2.CSV` dataset in the root directory.
4. Run the R scripts in sequence to analyze the heart disease dataset.

## Authors

- [Your Name] - [Your Email]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
