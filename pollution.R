# This is an R project for PS0002 DSAI (AY2020/21 Semester 2).

# Data are from McDonald and Schwing (1973), "Instabilities of Regression Estimates Relating Air Pollution to Mortality," Technometrics, 15, 463-481.

# Description of Variables:
# `Y`  Total Age Adjusted Mortality Rate
# `x1` Mean annual precipitation in inches
# `x2` Mean January temperature in degrees Fahrenheit
# `x3` Mean July temperature in degrees Fahrenheit
# `x4` Percent of 1960 SMSA population that is 65 years of age or over
# `x5` Population per household, 1960 SMSA
# `x6` Median school years completed for those over 25 in 1960 SMSA
# `x7` Percent of housing units that are found with facilities
# `x8` Population per square mile in urbanized area in 1960
# `x9` Percent of 1960 urbanized area population that is non-white
# `x10` Percent of families with income under 3,000 in 1960 urbanized area
# `x11` Percent employment in white-collar occupations in 1960 urbanized area
# `x12` Relative population potential of hydrocarbons, HC
# `x13` Relative pollution potential of oxides of nitrogen, NOx
# `x14` Relative pollution potential of sulfur dioxide, SO2
# `x15` Percent relative humidity, annual average at 1 p.m.

library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(datarium)
library(psych)
library(gridExtra)
library(factoextra)
library(e1071)

# <-------------------- Data Preparation -------------------->
read.table("pollution.txt", header=T) -> pollution

head(pollution)

pollution2 <- mutate(pollution, risk=ifelse(790 <= y & y <= 899, "low", ifelse(900 <= y & y <= 984, "medium", "high")))
# It is because Q1 = 899 and Q3 = 984 by direct calculatons
pollution2$risk <- as.factor(pollution2$risk)

#risk.low <- pollution2 %>% filter(risk %in% c("low"))
#risk.medium <- pollution2 %>% filter(risk %in% c("medium"))
#risk.high <- pollution2 %>% filter(risk %in% c("high"))

# <-------------------- Rename the variables -------------------->
pollution2 <- pollution2 %>% rename(age = y, 
                                    prep = x1, 
                                    jan.temp = x2, 
                                    jul.temp = x3, 
                                    older.65 = x4, 
                                    ppl.household = x5, 
                                    school.year = x6,
                                    housing.unit = x7,
                                    ppl.sqmile = x8,
                                    ppl.nonwhite = x9, 
                                    income = x10, 
                                    white.collar = x11, 
                                    hc = x12, 
                                    nox = x13, 
                                    so2 = x14, 
                                    rel.humidity = x15)
pollution <- pollution %>% rename(age = y, 
                                  prep = x1, 
                                  jan.temp = x2, 
                                  jul.temp = x3, 
                                  older.65 = x4, 
                                  ppl.household = x5, 
                                  school.year = x6,
                                  housing.unit = x7,
                                  ppl.sqmile = x8,
                                  ppl.nonwhite = x9, 
                                  income = x10, 
                                  white.collar = x11, 
                                  hc = x12, 
                                  nox = x13, 
                                  so2 = x14, 
                                  rel.humidity = x15)

pollution <- as_tibble(pollution)
pollution2 <- as_tibble(pollution2)

str(pollution2)
summary(pollution2)

# <-------------------- Histogram density plot -------------------->
ggplot(pollution, aes(age)) + geom_histogram(bins = 30, aes(y=..density..), colour = "black", fill = "white", na.rm = TRUE) + labs(x="Total Age Adjusted Mortality Rate", y="Density")  + geom_density(alpha = .2, fill = "pink")

# <-------------------- Histogram frequency plot -------------------->
h <- hist(pollution$age, xlab="Total Age Adjusted Mortality Rate", main="", ylim=c(0, 10), breaks=20)
text(h$mids, h$counts, labels=h$counts, adj=c(0.5, -0.5))
lines(density(pollution$age))


# <-------------------- Correlation plot -------------------->
corrplot(cor(pollution), type = "upper", method = "color",  addCoef.col = "black", number.cex = 0.5)

col2 <- colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                           "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                           "#4393C3", "#2166AC", "#053061"))
corrplot(cor(pollution), order = "hclust", addrect = 2, col = col2(50))

pairs.panels(pollution, method = "pearson", hist.col =  "steelblue", pch = 21, density = TRUE, ellipses = FALSE )


# <-------------------- Line plot -------------------->
pollution %>% ggplot(aes(x = ppl.nonwhite, y = age)) + geom_point() + geom_smooth() + labs(x="Percent of  population that is non-white (%)", y="Total Age Adjusted Mortality Rate")
# This shows that there is a clear relation between ppl.nonwhite and age


# <-------------------- Training and Testing Set Preparation -------------------->
training.idx <- sample(1: nrow(pollution2), size=nrow(pollution2) * 0.8)
train.data  <- pollution2[training.idx, ]
test.data <- pollution2[-training.idx, ]





# <-------------------- kNN Model (First Method) -------------------->
# All predictor variables are in linear order
model.knn <- train(age ~.,  data = pollution, method = "knn", trControl = trainControl("cv", number = 3), preProcess = c("center", "scale"), tuneLength = 10)
plot(model.knn)
model.knn$bestTune
predictions <- predict(model.knn, test.data)
RMSE(predictions, test.data$age) # compute the prediction error RMSE

# <-------------------- kNN Model (Second Method) -------------------->
# Significant predictor variables are in second order; otherwise, in linear order
model.knn.2 <- train(age ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity,  data = pollution, method = "knn", trControl = trainControl("cv", number = 3), preProcess = c("center", "scale"), tuneLength = 10)
plot(model.knn.2)
model.knn.2$bestTune
predictions <- predict(model.knn.2, test.data)
RMSE(predictions, test.data$age)

plot(test.data$age, predictions, main = "Prediction performance of kNN regression")
abline(0, 1, col="red") # add a reference line x=y





# <-------------------- LR Model (First Method) -------------------->
lmodel <- lm(age ~., data = train.data)
predictions <- predict(lmodel, test.data)
RMSE(predictions, test.data$age)
par(mfrow=c(2, 2))
plot(lmodel)

# Removing outliers
pollution2.1 <- polution2[-c(8, 34, 14), ]
training.idx.2 <- sample(1:nrow(pollution2.1), size=nrow(pollution2.1)*0.8)
train.data.2 <- pollution2.1[training.idx.2, ]
test.data.2 <- pollution2.1[-training.idx.2, ]

# <-------------------- LR Model (Second Method) -------------------->
lmodel.2 <- lm(age ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity, data = train.data.2)
summary(lmodel.2)
predictions <- predict(lmodel.2, test.data.2)
RMSE(predictions, test.data.2$age)

plot(test.data$age, predictions, main="Prediction performance of linear regression")
abline(0, 1, col = "red")
# Residual plot
par(mfrow=c(2, 2))
plot(lmodel.2)





# <-------------------- RF Model (First Method) -------------------->
tg <- data.frame(mtry = seq(2, 20, by =2))
rf1 <- train(age~., data = pollution, method = "rf", tuneGrid = tg)
rf1$results
predictions <- predict(rf1, test.data)
RMSE(predictions, test.data$age)

# <-------------------- RF Model (Second Method) -------------------->
rf2 <- train(age ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity, data = pollution, method = "rf", tuneGrid = tg)
rf2$results
rf2$finalModel
predictions <- predict(rf2, test.data)
RMSE(predictions, test.data$age)

plot(test.data$age, predictions, main="Prediction performance of Random Forest Model", ylab="Predictions RF")
abline(0, 1, col = "red")





# <-------------------- Classification: SVM -------------------->
svm.linear.2 <- svm(risk ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity + age, data = train.data, kernel = "linear")
svm.radial.2 <- svm(risk ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity + age, data = train.data, kernel = "radial")
svm.sigmoid.2 <- svm(risk ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity + age, data = train.data, kernel = "sigmoid")
svm.linear.1 <- svm(risk ~ prep+ jan.temp + jul.temp + older.65 + ppl.household + school.year + housing.unit + ppl.sqmile + ppl.nonwhite + white.collar + income + hc + nox + so2 + rel.humidity + age, data = train.data, kernel = "linear")
svm.radial.1 <- svm(risk ~ prep+ jan.temp + jul.temp + older.65 + ppl.household + school.year + housing.unit + ppl.sqmile + ppl.nonwhite + white.collar + income + hc + nox + so2 + rel.humidity + age, data = train.data, kernel = "radial")
svm.sigmoid.1 <- svm(risk ~ prep+ jan.temp + jul.temp + older.65 + ppl.household + school.year + housing.unit + ppl.sqmile + ppl.nonwhite + white.collar + income + hc + nox + so2 + rel.humidity + age, data = train.data, kernel = "sigmoid")

pred.svm.linear.2 <- predict(svm.linear.2, newdata=test.data)
pred.svm.radial.2 <- predict(svm.radial.2, newdata=test.data)
pred.svm.sigmoid.2 <- predict(svm.sigmoid.2, newdata=test.data)
pred.svm.linear.1 <- predict(svm.linear.1, newdata=test.data)
pred.svm.radial.1 <- predict(svm.radial.1, newdata=test.data)
pred.svm.sigmoid.1 <- predict(svm.sigmoid.1, newdata=test.data)

table(pred.svm.linear.1, test.data$risk)
table(pred.svm.linear.2, test.data$risk)
table(pred.svm.radial.1, test.data$risk)
table(pred.svm.radial.2, test.data$risk)
table(pred.svm.sigmoid.1, test.data$risk)
table(pred.svm.sigmoid.2, test.data$risk)

mean(pred.svm.linear.1 == test.data$risk)
mean(pred.svm.linear.2 == test.data$risk)
mean(pred.svm.radial.1 == test.data$risk)
mean(pred.svm.radial.2 == test.data$risk)
mean(pred.svm.sigmoid.1 == test.data$risk)
mean(pred.svm.sigmoid.2 == test.data$risk)





# <-------------------- Classification: Multi-class Logistic Regression -------------------->
# Fit the model
lg.model.2 <- nnet::multinom(risk ~ I(prep^2) + jan.temp + jul.temp + older.65 + ppl.household + I(school.year^2) + I(housing.unit^2) + ppl.sqmile + I(ppl.nonwhite^2) + white.collar + I(income^2) + hc + nox + I(so2^2) + rel.humidity, data = train.data)
lg.model.1 <- nnet::multinom(risk ~ prep + jan.temp + jul.temp + older.65 + ppl.household + school.year + housing.unit + ppl.sqmile + ppl.nonwhite + white.collar + income + hc + nox + so2 + rel.humidity, data = train.data)

# Make predictions
predicted.classes.2 <- lg.model.2 %>% predict(test.data)
predicted.classes.1 <- lg.model.1 %>% predict(test.data)

# Model accuracy
mean(predicted.classes.1 == test.data$risk)
mean(predicted.classes.2 == test.data$risk)

# Comparison table
table(predicted.classes.1, test.data$risk)
table(predicted.classes.2, test.data$risk)





# <-------------------- EXTRA -- K-MEANS CLUSTERING -------------------->

# <-------------------- Unsupervised Learning (k=3) -------------------->
pol <- scale(pollution)
k3 <- kmeans(pol, centers = 3, nstart = 25)
str(k3)
k3
table(pollution2$risk, k3$cluster)

# <-------------------- Unsupervised Learning (k=6) -------------------->
k6 <- kmeans(pol, centers = 6, nstart = 25)
str(k6)
k6


# WSS Plot
wcss <- function(k) { 
  kmeans(pol, k, nstart = 10 )$tot.withinss
} 
k.values <- 1:20 # compute and plot wss for k = 1 to k = 20 
wcss_k <- sapply(k.values, wcss) # apply wcss to all k values
plot(k.values, wcss_k, type="b", pch = 19, frame = FALSE, xlab="Number of clusters K", ylab="Total within-clusters sum of squares")
points(x=6, y=393, pch=21, col="red", cex=3, lwd=5)


# Comparison between k=3 and k=6

k4 <- kmeans(pol, centers=4, nstart=25)
k5 <- kmeans(pol, centers=5, nstart=25)

p1 <- fviz_cluster(k3, geom = "point", data = pollution) + ggtitle("k = 3")
p2 <- fviz_cluster(k6, geom = "point", data = pollution) + ggtitle("k = 6")
p3 <- fviz_cluster(k4, geom = "point", data = pollution) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point", data = pollution) + ggtitle("k = 5")

grid.arrange(p1, p2, p3, p4, nrow=2)

# Pairwise standard scatterplot
s1 <- fviz_cluster(k3, geom = "point", data = pollution[, c(9, 16)]) + ggtitle("k = 3")
s2 <- fviz_cluster(k6, geom = "point",  data = pollution[, c(9, 16)]) + ggtitle("k = 6")
grid.arrange(s1, s2, nrow = 1)

# Extract the clusters and add it to the initial data to summarize descriptive statistics at the cluster level:
pollution %>% mutate(Cluster = k6$cluster) %>% group_by(Cluster) %>% summarise_all("mean")


