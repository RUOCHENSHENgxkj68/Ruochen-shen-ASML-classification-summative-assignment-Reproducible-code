set.seed(2022)
data <- read.csv("C:/Users/yuhzh/Desktop/Classification/data/telecom.csv", header = TRUE)

summary(data)

Col <- seq(1, 20)
factorCol <- Col[-c(2, 5, 18, 19)]

for(i in factorCol){
  data[, i] <- as.factor(data[, i])
}

data <- na.omit(data)

summary(data)

library("car")

algorithms.name <- c("LDA", "QDA", "Logistic Regression", "k-Nearest Neighbor", "Naive Bayes", "Random Forest", "Support Vector Machine", "AdaBoost", "Deep Learning")
algorithms.accuracy <- rep(0, length(algorithms.name))
algorithms.tp <- rep(0, length(algorithms.name))
  
par(mfrow=c(2,2))
qqPlot(data$SeniorCitizen)

qqPlot(data$tenure)

qqPlot(data$MonthlyCharges)

qqPlot(data$TotalCharges)

library(funModeling) 
library(tidyverse) 
library(Hmisc)

freq(data)
freq(data$Churn)
plot_num(data)

data_prof <- profiling_num(data)
data_prof

new.data <- data
for(i in factorCol){
  new.data[, i] <- as.numeric(new.data[, i])
}
new.data$Churn <- new.data$Churn - 1

head(new.data)

train.index <- sample(1:nrow(new.data), nrow(new.data)*0.80, replace=F)
train <- new.data[train.index, ]
test <- new.data[-train.index, ]

nrow(train)

nrow(test)

library(MASS)

tel.lda <- lda(Churn~., data = train, family = binomial)
pred <- predict(tel.lda, test, family = binomial)
result <- table(pred$class, test$Churn)

accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[1] <- accuracy
algorithms.tp[1] <- result[2, 2] / sum(test$Churn==1)

tel.qda <- qda(Churn~., data = train, family = binomial)
pred <- predict(tel.qda, test, family = binomial)
result <- table(pred$class, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[2] <- accuracy
algorithms.tp[2] <- result[2, 2] / sum(test$Churn==1)

tel.glm <- glm(Churn~., data = train, family = binomial)
prob <- predict(tel.glm, test)
pred <- ifelse(prob > 0.5, 1, 0)
result <- table(pred, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[3] <- accuracy
algorithms.tp[3] <- result[2, 2] / sum(test$Churn==1)

library(class)

col <- seq(1, 19)

accuracy.knn = seq(1, 30)
tp.knn = seq(1, 30)
for(i in 1:30){
  pred <- knn(as.matrix(train[, col]), as.matrix(test[, col]), train[, 20], k=i)
  result <- table(pred, test$Churn)
  accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
  
  accuracy.knn[i] <- accuracy
  tp.knn[i] <- result[2, 2] / sum(test$Churn==1)
}
accuracy.knn
algorithms.accuracy[4] <- min(accuracy.knn)
algorithms.tp[4] <- tp.knn[which.min(accuracy.knn)]
par(mfrow=c(1,1))
plot(1:30, accuracy.knn, xlab = 'K', ylab = 'Accuracy on validation set')

library(e1071)
nb.model <- naiveBayes(Churn~., data = train)
nb_predict <- predict(nb.model, newdata = test)
result <- table(nb_predict, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[5] <- accuracy
algorithms.tp[5] <- result[2, 2] / sum(test$Churn==1)


library(randomForest)
random.train <- train
random.train$Churn <- as.factor(random.train$Churn)
tel.random <- randomForest(Churn~., data=random.train)
pred <- predict(tel.random, newdata = test)
result <- table(pred, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[6] <- accuracy
algorithms.tp[6] <- result[2, 2] / sum(test$Churn==1)


library(e1071)
model <- svm(Churn~., data = random.train)
svm.pred<-predict(model, test[, !names(test)%in%c("Churn")])
result <- table(svm.pred, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[7] <- accuracy
algorithms.tp[7] <- result[2, 2] / sum(test$Churn==1)


library(adabag)
library(pROC)

tel.adaboost <- boosting(Churn~., data = random.train, boos=TRUE, mfinal=100)
pred <- predict(tel.adaboost, newdata = test)$class
result <- table(pred, test$Churn)
accuracy <- (result[1, 1] + result[2, 2]) / sum(result)
accuracy
algorithms.accuracy[8] <- accuracy
algorithms.tp[8] <- result[2, 2] / sum(test$Churn==1)



library(keras)
library(dplyr)
library(ggplot2)
library(purrr)

pca.data <- new.data[, -c(20)]
tel.pca <- prcomp(scale(pca.data), center = TRUE)
names(tel.pca)
summary(tel.pca)
x.var <- tel.pca$sdev ^ 2
x.pvar <- x.var/sum(x.var)
cumsum(x.pvar)[16]
pca.data <- tel.pca$x[, 1:16]

train <- pca.data[train.index, ]
test <- pca.data[-train.index, ]


train_x <- array(as.matrix(train), dim = c(dim(train)[1], dim(train)[2]))
test_x <- array(as.matrix(test), dim = c(dim(test)[1], dim(test)[2]))

train_y <- to_categorical(new.data[train.index, ]$Churn, 2)
test_y <- to_categorical(new.data[-train.index, ]$Churn, 2)

set.seed("1023")

lr <- c(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5)
batch.size <- c(8, 16, 32, 64, 128, 256, 512, 1024)
lr.accuracy <- rep(0, length(lr))
batch.size.accuracy <- rep(0, length(batch.size))

for(i in 1:length(lr)){
model <- keras_model_sequential()

model %>% 
  layer_dense(units = 64, input_shape = 16) %>% 
  layer_dropout(rate=0.2)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 128) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 64) %>% 
  layer_dropout(rate=0.2)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 2) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = lr[i]),
  metrics = c('accuracy')
)

model %>% fit(train_x, train_y, epochs = 100, batch_size = 128, validation_split = 0.2, callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = 20, restore_best_weights = TRUE)))
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)
lr.accuracy[i] <- loss_and_metrics[2]
}

par(mfrow=c(1,1))
best.lr <- lr[which.max(lr.accuracy)]
plot(lr, lr.accuracy, xlab = "Learning rate", ylab = "Accuracy on Validation dataset")

for(i in 1:length(batch.size)){
  model <- keras_model_sequential()
  
  model %>% 
    layer_dense(units = 64, input_shape = 16) %>% 
    layer_dropout(rate=0.2)%>%
    layer_activation(activation = 'relu') %>% 
    layer_dense(units = 128) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_dense(units = 64) %>% 
    layer_dropout(rate=0.2)%>%
    layer_activation(activation = 'relu') %>% 
    layer_dense(units = 2) %>% 
    layer_activation(activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(learning_rate = learning),
    metrics = c('accuracy')
  )
  
  model %>% fit(train_x, train_y, epochs = 100, batch_size = batch.size[i], validation_split = 0.2, callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = 20, restore_best_weights = TRUE)))
  loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = batch.size[i])
  batch.size.accuracy[i] <- loss_and_metrics[2]
}

par(mfrow=c(1,2))
best.lr <- lr[which.max(lr.accuracy)]
plot(lr, lr.accuracy, xlab = "Learning rate", ylab = "Accuracy on Validation dataset")

best.batchsize <- batch.size[which.max(batch.size.accuracy)]
plot(batch.size, batch.size.accuracy, xlab = "Batch Size", ylab = "Accuracy on Validation dataset")


model <- keras_model_sequential()

model %>% 
  layer_dense(units = 64, input_shape = 16) %>% 
  layer_dropout(rate=0.2)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 128) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 64) %>% 
  layer_dropout(rate=0.2)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 2) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = best.lr),
  metrics = c('accuracy')
)

model %>% fit(train_x, train_y, epochs = 100, batch_size = best.batchsize, validation_split = 0.2, callbacks = list(callback_early_stopping(monitor = "val_accuracy", patience = 20, restore_best_weights = TRUE)))
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = best.batchsize)


best.batchsize
best.lr

prob <- predict(model, test_x)
pred <- ifelse(prob[, 1] > prob[, 2], 1, 0)
target_y <- ifelse(test_y[, 1] > test_y[, 2], 1, 0)
result <- table(pred, target_y)
algorithms.accuracy[9] <- accuracy
algorithms.tp[9] <- result[2, 2] / sum(target_y==1)

par(mfrow=c(1,2))
plot(1:9, algorithms.accuracy, xlab = "Algorithms", ylab = "Accuracy on Validation Dataset")
text(1:9, algorithms.accuracy, algorithms.name, cex=0.6, pos=4, col="red")
plot(1:9, algorithms.tp, xlab = "Algorithms", ylab = "True Positive Rates on Validation Dataset")
text(1:9, algorithms.tp, algorithms.name, cex=0.6, pos=4, col="red")
