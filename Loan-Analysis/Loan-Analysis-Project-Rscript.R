##########################################################
#  Name: Roland DePratti
#  Course: Data-513
#  Date: 11/30/2020
#  Course Project 4
##########################################################
# 1a. load packages needed later
# Read files
##########################################################
library(caret)
library(ggplot2)
library(rattle)
library(psych)
set.seed(123)
setwd("D:/Education/DataScience/CCSU-Data-Science/Data-Mining-And-Predictive-Analytics/Data-513/Week-15")
loans.train <- read.csv("loans_train_z_3_3x.csv")
summary(loans.train)
loans.test <- read.csv("loans_test_z.csv")
summary(loans.test)
####################################
# 1b. Data Preparation
####################################
loans.train$Default <-as.factor(loans.train$Default)
loans.train$term <-as.factor(loans.train$term)
loans.train$X <- NULL
summary(loans.train)
loans.test$Default <-as.factor(loans.test$Default)
loans.test$term <-as.factor(loans.test$term)
loans.test$X <- NULL
summary(loans.test)
#############################################################
# 1c. Check datasets
#############################################################
summary(loans.test$Default)
summary(loans.train$Default)
#############################################################
# 2a. CART Model
#############################################################
seeds.cart = c(123,128,132,135,139,141,145,149,151,154,160)
TC.cart <- trainControl(method = "CV", number = 10, seeds=seeds.cart)
fit.CART <- train(Default ~ ., data = loans.train, 
                  method = "rpart", trControl = TC)
fit.CART$resample
testsetpreds.CART <- predict(fit.CART, loans.test)
table(loans.test$Default, testsetpreds.CART)
#############################################################
# 2b. Treebag Model
#############################################################
seeds = c(123,128,132,135,139,141,145,149,151,154,160)
TC.bag <- trainControl(method = "CV", number = 10, seeds = seeds)
set.seed(193)
fit.bag <- train(Default ~ ., data = loans.train, 
                  method = "treebag", trControl = TC.bag)
fit.bag$resample
testsetpreds.bag <- predict(fit.bag, loans.test)
table(loans.test$Default, testsetpreds.bag)
#############################################################
# 2c. Adaboost Model
#############################################################
set.seed(123)
fit.boost <- train(Default ~ ., data = loans.train, 
                 method = "adaboost", trControl = TC)
fit.boost$resample
testsetpreds.boost <- predict(fit.boost, loans.test)
table(loans.test$Default, testsetpreds.boost)
#############################################################
# 2d. Random Forest Model
#############################################################
set.seed(193)
seeds.rf <- vector(mode= "list", length = 11)
for(i in 1:10) 
   { seeds.rf[[i]]<- sample.int(n=1000, 3)}
seeds.rf[[11]] <- sample.int(n=1000, 1)
TC.rf <- trainControl(method = "CV", number = 10, seeds = seeds.rf)
fit.rf <- train(Default ~ ., data = loans.train, 
                 method = "rf", trControl = TC.rf)
fit.rf$resample
testsetpreds.rf <- predict(fit.rf, loans.test)
table(loans.test$Default, testsetpreds.rf)
#############################################################
# 2e. C5.0 Model
#############################################################
seeds.c5 <- vector(mode= "list", length = 11)
for(i in 1:10) 
{ seeds.c5[[i]]<- sample.int(n=1000, 4)}
seeds.c5[[11]] <- sample.int(n=1000, 1)
TC.c5 <- trainControl(method = "CV", number = 10, seeds = seeds.c5)
set.seed(123)
fit.C5 <- train(Default ~ ., data = loans.train, 
                 method = "C5.0", trControl = TC.c5, fitbest=FALSE, returnData = TRUE)
fit.C5$resample
testsetpreds.C5 <- predict(fit.C5, loans.test)
table(loans.test$Default, testsetpreds.C5)
fit.C5$predictors
predictors(fit.C5)
#############################################################
# 3. Apply C5.0 Model to Holdout Dataset
#############################################################
set.seed(123)
setwd("D:/Education/DataScience/CCSU-Data-Science/Data-Mining-And-Predictive-Analytics/Data-513/Week-15")
loans.validation <- read.csv("loans_proj4_validation")
loans.validation$grade <-as.factor(loans.validation$grade)
loans.validation$term <-as.factor(loans.validation$term)
loans.validation$Index <-as.factor(loans.validation$Index)
summary(loans.validation)
#############################################################
# 3b. Standardize Data
#############################################################
set.seed(123)
preprocess.validation.z <- preProcess(loans.validation[1:6], 
                                 method=c("center", "scale") )
preprocess.validation.z
loans.validation.z <- predict(preprocess.validation.z, loans.validation[1:6] )
describe(loans.validation.z)
summary(loans.validation.z)
#############################################################
# 3c. Make Predictions on Holdout
#############################################################
# add default dummy to allow for use of predict function
loans.validation.z$Default <- as.factor(0)
loans.validation.sz <- loans.validation.z[order(loans.validation.z$Index),]
valsetpreds.C5 <- predict(fit.C5, loans.validation.sz)
table(loans.validation.sz$Default, valsetpreds.C5)
# add prediction to original dataset
loans.validation.sz$Default <- valsetpreds.C5
# summarize data to look for patterns
aggregate(loans.validation.sz$Default, by=list(Default=loans.validation.sz$Default,Grade=loans.validation.sz$grade), FUN=function(x){NROW(x)})
# plot pattern showing where model predicts defaults by grade
ggplot(loans.validation.sz, aes(grade)) + geom_bar(aes(fill = Default), position = "stack") + 
  xlab("Grade") + ylab("Count") + ggtitle("Bar Chart of Loan Grade with Predicted Default Overlay")
# prepare output
predictions <- as.data.frame(valsetpreds.C5)
names(predictions)[names(predictions) == "valsetpreds.C5"] <- "Predictions"
setwd("D:/Education/DataScience/CCSU-Data-Science/Data-Mining-And-Predictive-Analytics/Data-513/Week-15")
write.table(predictions, "Predictions.csv", row.names=F)	
