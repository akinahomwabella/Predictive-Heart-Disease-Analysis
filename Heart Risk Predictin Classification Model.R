install.packages("ISLR")

install.packages("dplyr")

library(ISLR)

library(ggplot2)

library(dplyr)




Hdat2<-read.table("HEART2.CSV", sep=",", header=TRUE)
Hdat2


str(Hdat2)

attach(Hdat2)

barplot(table(target)) 

barplot(table(sex)) 

tgt= ifelse( target== "yes", 0, 1)

glm1=glm(tgt~age+sex+chol,data=Hdat2 ,family=binomial )
summary(glm1)

Age=45

male=1

chol=202

L=-5.954+0.063*Age+1.676*male+0.0047*chol

P45=exp(L)/(1+exp(L))

P45*100

Age=55

male=1

chol=202

L55=-5.954+0.063*Age+1.676*male+0.0047*chol

P55=exp(L55)/(1+exp(L55))

P55*100


glm.probs=predict(glm1,type="response")

glm.probs[1:10]

?predict


glm.pred=rep("No" ,304)
glm.pred[glm.probs >.5]="Yes"
table(glm.pred,tgt)
length(glm.pred)
length(tgt)

(87+118)/304*100

##############################

set.seed(2)
train=sample(1: nrow(Hdat2), 152)

H.test=Hdat2[-train ,]
H.train=Hdat2[train ,]



tgt2= ifelse( H.train$target== "yes", 1, 0)

library(MASS)
lda.fit2=lda(tgt2~H.train$age+H.train$sex+H.train$chol,data=H.train)
lda.fit2

length(H.train$age)

plot(lda.fit2)


lda.pred=predict(lda.fit2, H.test)
names(lda.pred)

lda.class=lda.pred$class
table(lda.class ,H.test$target)

length(lda.class)
length(H.test$target)

(26+47)/(26+47+46+33)*100


########################

qda.fit2=qda(tgt2~age+sex+chol,data=H.train)
qda.fit2


qda.class=predict(qda.fit2, H.test)$class
table(qda.class, H.test$target)

(48+58)/(48+58+24+22)*100

########################
# Ensure the dataset is correctly attached and its structure is suitable for operations
attach(Hdat2)

# Scaling the data excluding certain columns (columns 3 and 15)
St.X = scale(Hdat2[, -c(3, 15)])

# Convert 'target' to a factor if it's not already
targetf = as.factor(Hdat2$target)

# Indices for test set
test_indices = 1:100

# Prepare training and test data
train.X = St.X[-test_indices,]  # Training data are all except the first 100
test.X = St.X[test_indices,]    # Test data are the first 100

# Ensure that train.Y and test.Y are assigned correctly
train.Y = targetf[-test_indices]  # Training labels corresponding to train.X
test.Y = targetf[test_indices]    # Test labels corresponding to test.X

# Set seed for reproducibility
set.seed(1)

# Perform the KNN classification
knn.pred = knn(train.X, test.X, train.Y, k = 1)

# Create a table of predictions versus actual test labels
result_table = table(knn.pred, test.Y)
print(result_table)


?sample


##########################################################
### Data SAMPLING ####
install.packages("caret")

library(caret)

set.seed(101)
split = createDataPartition(tgt, p = 0.80, list = FALSE)
train_data = Hdat2[split,]
test_data = Hdat2[-split,]

#error metrics -- Confusion Matrix
err_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  False_positive_rate =(FP)/(FP+TN)
  False_negative_rate =(FN)/(FN+TP)
  print(paste("Precision value of the model: ",round(precision,2)))
  print(paste("Accuracy of the model: ",round(accuracy_model,2)))
  print(paste("Recall value of the model: ",round(recall_score,2)))
  print(paste("False Positive rate of the model: ",round(False_positive_rate,2)))
  print(paste("False Negative rate of the model: ",round(False_negative_rate,2)))
  print(paste("f1 score of the model: ",round(f1_score,2)))
}

str(train_data)
str(test_data)

attach(train_data)

tgt= ifelse( target== "yes", 0, 1)


# 1. Logistic regression
logit_m =glm(formula=tgt~ age+sex+chol ,data =train_data ,family='binomial')
summary(logit_m)

logit_P = predict(logit_m , newdata = test_data[-15] ,type = 'response' )
logit_P <- ifelse(logit_P > 0.5,1,0) # Probability check
CM= table(test_data[,13] , logit_P)
print(CM)
err_metric(CM)

#ROC-curve using pROC library
library(pROC)
roc_score=roc(test_data[,15], logit_P) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

