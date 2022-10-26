#___________________________________________________________________________________________
#-------------------------------------Code from Yvonne--------------------------------------
#___________________________________________________________________________________________

#___________________________________________________________________________________________
###Slide 12-13###
#___________________________________________________________________________________________

"Naive-Bayes model"
install.packages('e1071', dependencies=TRUE)
library(e1071)
df <- read.csv("WineQT.csv")
df <- df[c(2,3,10,11,12)]
"take 80 percent of the dataset as training data"
train = sample(1:nrow(df), size = 0.8*nrow(df))
train_df = df[train,]
test_df = df[-train,]
nrow(train_df)
nrow(test_df)
library('caret')
if(!require('caret')) {
  install.packages('e1071', dependencies=TRUE)
}
summary(df)
names(df)
nb.fit <- naiveBayes(quality~. , data = train_df)
nb.fit
"predict the result with naive bayes model"
y_pred <- predict (nb.fit , test_df)
y_pred

confusion_matrix = confusionMatrix(y_pred,as.factor(test_df$quality))
confusion_matrix



#___________________________________________________________________________________________
#-------------------------------------Code from Aditya--------------------------------------
#___________________________________________________________________________________________

################################################################################
## Random Forest Classification Model
################################################################################

#___________________________________________________________________________________________
###Slide 15-17###
#___________________________________________________________________________________________

library(readr)
library(dplyr)

wineDf <- read_csv("C:/Users/HP/OneDrive/Desktop/UT Austin MSBA/Intro to Machine Learning/Project/WineQT.csv")
head(wineDf)

str(wineDf)

# We don't need Id values as part of the dataset
wineDf = subset(wineDf,select= -c(Id))

# We will first check if there are any null/empty or NA values present in the dataset
any(is.na(wineDf))  #This returns fall so there are no NA values in the dataframe

#___________________________________________________________________________________________
###Slide-4
#___________________________________________________________________________________________

#We plot the percentage of missing values as well for each column
library(DataExplorer)
plot_missing(wineDf)

#Check the data types of the dataframe
str(wineDf)


colnames(wineDf)
attach(wineDf)

library(randomForest)

#Convert the quality column into a factor for classification 
wineDf$quality <- as.factor(wineDf$quality)
str(wineDf)

#Change column names to cater to random forest model
names(wineDf)[1] <- 'fixed_acidity'
names(wineDf)[2] <- 'volatile_acidity'
names(wineDf)[3] <- 'citric_acid'
names(wineDf)[4] <- 'residual_sugar'
names(wineDf)[6] <- 'free_sulphur_dioxide'
names(wineDf)[7] <- 'total_sulphur_dioxide'

str(wineDf)


library(ggplot2)
#Set seed for the split
set.seed(123)

#Create training and test samples for the model
idx <- sample(1:nrow(wineDf), nrow(wineDf)*0.7, replace=TRUE)
train <- wineDf[idx,]
test <- wineDf[-idx,]

#___________________________________________________________________________________________
###Slide-15
#___________________________________________________________________________________________
################################################################################
## Create Model with different number of trees
################################################################################

modelRandomforest <- function(numTree,variableSize)
{
  set.seed(123)
  
  model.rf <- randomForest(quality ~ ., train,proximity=TRUE,ntree=numTree,mtry=variableSize)
  model.rf$err.rate
  
  return((model.rf))
}

#___________________________________________________________________________________________
###Slide 16
#___________________________________________________________________________________________
################################################################################
## Plot the OOB error rate graph across the trees; 
################################################################################

plottingModel <- function(model)
{
  set.seed(123)
  
  oob.error.rate  <- data.frame(Trees=rep(1:nrow(model.rf$err.rate),times=7,mtry=5),
                                Type=rep(c("OOB","3","4","5","6","7","8"),each=nrow(model.rf$err.rate)),
                                Error=c(model.rf$err.rate[,"OOB"],model.rf$err.rate[,"3"],model.rf$err.rate[,"4"],
                                        model.rf$err.rate[,"5"],model.rf$err.rate[,"6"],model.rf$err.rate[,"7"],model.rf$err.rate[,"8"]))
  
  
  ggplot(data=oob.error.rate,aes(x=Trees,y=Error)) + geom_line(aes(color=Type))
}

#Check the model OOB for variety of num trees from 500 to 5000
tree.values <- vector(length=10)
count=1

for(i in seq(500,5000,500))
{
  cat ('Doing modelling with number of trees:  ',i,'\n')
  model.rf = modelRandomforest(i,3)
  tree.values[count]= model.rf$err.rate[nrow(model.rf$err.rate),1]
  count=count+1
  print(model.rf)
}
tree.values

#Pick the tree which has the least OOB error
finalTree = which(tree.values==min(tree.values))*500
final.model.rf = modelRandomforest(finalTree,3)
final.model.rf
plottingModel(final.model.rf)

#Slide-17
#Plot variable importance graph for the dataset
varImpPlot(final.model.rf)

#___________________________________________________________________________________________
###Slide-17
#___________________________________________________________________________________________
################################################################################
## Find the optimum number of variables to be splitted across for the dataset
################################################################################

oob.values <- vector(length=11)

for(i in 1:11){
  set.seed(123)
  temp.model <- randomForest(quality~.,data=train,mtry=i,ntree=finalTree)
  oob.values[i]= temp.model$err.rate[nrow(temp.model$err.rate),1]
}


oob.values
minValue = min(oob.values)
minValue
minValueIdx = which(oob.values==min(oob.values))[1]
minValueIdx
#Variables of 3 is the best out of the lot

#Final Decision tree
model.rf =modelRandomforest(finalTree,minValueIdx)
model.rf
plottingModel(model.rf)

str(test)
#___________________________________________________________________________________________
###Slide-17
#___________________________________________________________________________________________
################################################################################
## Predict using the final random forest model
################################################################################

rf.pred <- predict(model.rf, test[, -12])
head(rf.pred)
summary(rf.pred)

#Get the confusion matrix for the final predicted values
library(caret)
confusionMatrix(rf.pred,test$quality)

#___________________________________________________________________________________________
###Slide-18
#___________________________________________________________________________________________
accuracyDf <- data.frame(name=c("Random Forest","Knn","Naive Bayes"),value = c(64.64,62.39,59.83))
library(ggplot2)
plot<-ggplot(accuracyDf,
             aes(name,value)) +
  geom_bar(stat = "identity")+
  geom_text(aes(label = signif(value)), nudge_y = 3) + ggtitle("ClassifciModel Accuracy Score") +
  xlab("Models") + ylab("Accuracy Score") + theme(plot.title = element_text(color = "red")) + theme(plot.title = element_text(face = "italic")) + theme(axis.title.x = element_text(colour = "blue"),axis.title.y = element_text(colour = "red")) + theme(axis.title.x = element_text(face = "bold"),axis.title.y = element_text(face = "bold"))  
plot

#___________________________________________________________________________________________
#------------------------------------Code from Nevin----------------------------------------
#___________________________________________________________________________________________
data = read.csv("WineQT.csv")

#___________________________________________________________________________________________
####Slide 20####
#___________________________________________________________________________________________

all_variables <- lm(quality ~., data = data)  
summary(all_variables)
plot(all_variables)
mean(all_variables$residuals^2) #mse = .4056305
plot(data$quality, all_variables$residuals, xlab = "Wine Quality", ylab = "Residuals")

#___________________________________________________________________________________________
###Slide 21
#___________________________________________________________________________________________

mult.reg <- lm(data$quality ~ data$volatile.acidity
               + data$sulphates + data$alcohol + data$chlorides + data$total.sulfur.dioxide + data$pH)
summary(mult.reg)
plot(mult.reg)
mean(mult.reg$residuals^2)
plot(data$quality, mult.reg$residuals)

#___________________________________________________________________________________________
###Slide 23####
#___________________________________________________________________________________________

set.seed(1)
library(glmnet)
training_size <- dim(data)[1]/2
train <- sample(1:dim(data)[1], training_size)
test <- -train

train_data <- data[train, ]
test_data <- data[test, ]
train_matrix2 <- model.matrix(quality ~., data = train_data)
test_matrix2 <- model.matrix(quality ~., data = test_data)

cv_ridge <- cv.glmnet(train_matrix2, train_data[, "quality"], alpha = 0)
best_lambda <- cv_ridge$lambda.min
best_lambda
ridge <- glmnet(train_matrix2, train_data$quality, alpha = 0, lamdba = best_lambda, thresh = 1e-12)
pred_ridge <- predict(ridge, newx = test_matrix2, s = best_lambda)
ridge_mse <- mean((pred_ridge-test_data[, "quality"])^2)
ridge_mse 

#___________________________________________________________________________________________
####Slide 24####
#___________________________________________________________________________________________

cv_lasso <- cv.glmnet(train_matrix2, train_data[, "quality"], alpha = 1)
best_lambda <- cv_lasso$lambda.min
best_lambda
lasso <- glmnet(train_matrix2, train_data$quality, alpha = 1, lamdba = best_lambda, thresh = 1e-12)
pred_lasso <- predict(lasso, newx = test_matrix2, s = best_lambda)
lasso_mse <- mean((pred_lasso-test_data[, "quality"])^2)
lasso_mse
plot(lasso, xvar = 'lambda')
plot(cv_lasso)
LamL = cv_lasso$lambda.1se
coef.L = predict(cv_lasso,type="coefficients",s=LamL)
plot(cv_lasso$lambda,sqrt(cv_lasso$cvm),main="Lasso CV (k=10)",xlab="lambda",ylab = "RMSE",col=4,type="b",cex.lab=1.2, xlim = c(0, 1))
predict(lasso, s = best_lambda, type = "coefficients")

LamR = cv_ridge$lambda.1se
plot(cv_ridge$lambda,sqrt(cv_ridge$cvm),main="Ridge CV (k=10)",xlab="lambda",ylab = "RMSE",col=4,type="b",cex.lab=1.2, xlim = c(0,1))
abline(v=LamR,lty=2,col=2,lwd=2)
coef.R = predict(cv_ridge,type="coefficients",s=LamR)
plot(abs(coef.R[2:20]),abs(coef.L[2:20]),ylim=c(0,1),xlim=c(0,1))

#___________________________________________________________________________________________
#--------------------------------------Code from Amrit--------------------------------------
#___________________________________________________________________________________________

#___________________________________________________________________________________________
###Slide 26####Regression Trees
#___________________________________________________________________________________________
#load up the file
file_load <- read.csv("WineQT.csv")

#load library
library(tree)
library(rpart)
library(randomForest)
library(gbm)
library(ggplot2)
set.seed(11)
##Create Training and test sets

#select sample set
sample.select = sample(dim(file_load)[1], dim(file_load)[1]/2)
training.set = file_load[sample.select, ]
testing.set = file_load[-sample.select, ]

#training the tree
regression.tree = tree(quality ~ ., data = training.set)
plot(regression.tree)
text(regression.tree)
summary(regression.tree)

#prediction and MSE
predicting.regresiontree = predict(regression.tree, testing.set)


#___________________________________________________________________________________________
#Use cross-validation in order to determine the optimal level of tree complexity. 
crossvalidation.tree = cv.tree(regression.tree, FUN = prune.tree)
par(mfrow = c(1, 2))
#min(crossvalidation.tree)
#plot(crossvalidation.tree$size, crossvalidation.tree$dev)
plot(crossvalidation.tree$size, crossvalidation.tree$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")
#the optimal size seems to be 8 here

#pruning the original regression tree
pruned.tree = prune.tree(regression.tree, best = 8)
par(mfrow = c(1, 1))
plot(pruned.tree)
text(pruned.tree)

summary(pruned.tree)

#predicting for Pruned tree
predicting.prunedtree = predict(pruned.tree, testing.set)
#error MSE
Error_MSE_pruned <- mean((testing.set$quality - predicting.prunedtree)^2)

#___________________________________________________________________________________________
###Slide 27####Random Forest
#___________________________________________________________________________________________

#Random Forest Default for Benchmarking
rf.wine = randomForest(quality ~ ., data = training.set)
rf.pred = predict(rf.wine, testing.set)
rf.error=mean((testing.set$quality - rf.pred)^2)
rf.error

#___________________________________________________________________________________________
#optimizing for best Mtry Selection
bestmtry <- tuneRF(file_load[,-13],file_load[,13],stepFactor = 1.5, trace=T, plot= T ) 

#___________________________________________________________________________________________
#optimizing for number of trees
tree.number = c(200,400,500,1000,2000,3000,5000,6000,7000)
length.tree.number = length(tree.number)
train.errors.tree = rep(NA, length.tree.number)
test.errors.tree = rep(NA, length.tree.number)


for (i in 1:length.tree.number) {
  random.wine = randomForest(quality ~ ., data = training.set,mtry=6, ntree = tree.number[i])
  train.pred = predict(random.wine, training.set, n.trees = tree.number[i])
  test.pred = predict(random.wine, testing.set, n.trees = tree.number[i])
  train.errors.tree[i] = mean((training.set$quality - train.pred)^2)
  test.errors.tree[i] = mean((testing.set$quality - test.pred)^2)
}

plot(tree.number, test.errors.tree, type = "b", xlab = "tree number", ylab = "Train MSE", col = "red", pch = 20)
lines(tree.number,test.errors.tree,type = "b",col="blue", pch = 20)

min(test.errors.tree)
tree.number[which.min(test.errors.tree)]

#___________________________________________________________________________________________
#Default Random Forest for Benchmarking
rf.wine = randomForest(quality ~ ., data = training.set)
rf.pred = predict(rf.wine, testing.set)
rf.error=mean((testing.set$quality - rf.pred)^2)
rf.error

plot(rf.wine,main="Number of trees vs MSE for different RF models ")

#___________________________________________________________________________________________
#Random Forest with high set Ntree and Mtry
rf.wine2 = randomForest(quality ~ ., data = training.set,maxnodes=24,ntree=5000, mtry=6)
rf.pred2 = predict(rf.wine2, testing.set)
rf.error2=mean((testing.set$quality - rf.pred2)^2)
rf.error2

points(1:5000,rf.wine2$mse,col="red",type="l")

#___________________________________________________________________________________________
#Random Forest with optimized Ntree and Mtry
rf.wine3 = randomForest(quality ~ ., data = training.set,ntree=1000, mtry=4,importance = TRUE)
rf.pred3 = predict(rf.wine3, testing.set)
rf.error3=mean((testing.set$quality - rf.pred3)^2)
rf.error3

points(1:1000,rf.wine3$mse,col="green",type="l")


#___________________________________________________________________________________________
# Get variable importance from the model fit
ImpData <- as.data.frame(randomForest::importance(rf.wine3))
ImpData$Var.Names <- row.names(ImpData)

ggplot(ImpData, aes(x=Var.Names, y=`%IncMSE`)) +
  geom_segment( aes(x=Var.Names, xend=Var.Names, y=0, yend=`%IncMSE`), color="skyblue") +
  geom_point(aes(size = IncNodePurity), color="blue", alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    legend.position="bottom",
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )

#___________________________________________________________________________________________
###Slide 28####Boosting
#___________________________________________________________________________________________

##Boosting Default###
boost.winedata <- gbm(quality ~ ., data = training.set,
                      distribution = "gaussian", n.trees = 1000,
                      interaction.depth = 4, shrinkage = 0.2, verbose = F)
yhat.boost <- predict(boost.winedata,
                      newdata = testing.set, n.trees = 1000)
MSE_boosting_shrinkage=mean((yhat.boost - testing.set$quality)^2)

#___________________________________________________________________________________________
#Boosting with varying lambda
#boosting on the training set with 1,000 trees for a range of values of the shrinkage parameter λ. Produce a plot with#different shrinkage values on the x-axis and the corresponding
#training set MSE on the y-axis.

pows = seq(-10, -0.2, by = 0.1)
lambdas = 10^pows
length.lambdas = length(lambdas)
train.errors = rep(NA, length.lambdas)
test.errors = rep(NA, length.lambdas)


for (i in 1:length.lambdas) {
  boost.wine = gbm(quality ~ ., data = training.set, distribution = "gaussian", 
                   n.trees = 1000, shrinkage = lambdas[i])
  train.pred = predict(boost.wine, training.set, n.trees = 1000)
  test.pred = predict(boost.wine, testing.set, n.trees = 1000)
  train.errors[i] = mean((training.set$quality - train.pred)^2)
  test.errors[i] = mean((testing.set$quality - test.pred)^2)
}

plot(lambdas, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", 
     col = "red", pch = 20, main="Plot Boosting MSE vs Lambda")
lines(lambdas,test.errors,type = "b",col="blue", pch = 20)

min(test.errors)
lambdas[which.min(test.errors)]

summary.gbm(boost.winedata)

#___________________________________________________________________________________________