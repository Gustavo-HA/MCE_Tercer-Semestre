library("ISLR")

college = College[, -c(3,4)] # Quitamos Accept y Enroll por data leakage

summary(college$Apps)

#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      81     776    1558    3002    3624   48094 

###### Inciso a) #######
set.seed(1)
train = sample(1:nrow(college), nrow(college)*0.5)
test = -train
college.train = college[train, ]
college.test = college[test, ]


##### Inciso b) #####
# Ajustamos un modelo de regresión lineal para predecir Apps
lm.fit = lm(Apps ~ ., data = college.train)     
summary(lm.fit)

# ***:
# - F. Undergrad
# - Room.Board

# **:
# - Expend

# *:
# - Grad.Rate

# .:
# - perc.alumni


# Predecimos en el conjunto de test y error de prueba
lm.pred = predict(lm.fit, newdata = college.test)
mean((lm.pred - college.test$Apps)^2) # MSE = 2551734
rmse = sqrt(mean((lm.pred - college.test$Apps)^2))
rmse # RMSE = 1597.4

##### Inciso c) #####
##### Ajustamos un modelo de regresión ridge
library("glmnet")

x = model.matrix(Apps ~ ., college.train)[, -1]
y = college.train$Apps  

val_lambda = 10^seq(5, -1, length = 50)

# Kfold cross-validation para elegir lambda
cv.ridge = cv.glmnet(x, y, lambda = val_lambda, nfolds = 5, alpha = 0)

bestlam_ridge = cv.ridge$lambda.min 
cat("Mejor lambda para ridge:", bestlam_ridge, "\n") # 12.06793

bestlam2_ridge = cv.ridge$lambda.1se
cat("Lambda 1se para ridge:", bestlam2_ridge, "\n") # 4498.433

par(mfrow = c(1, 1), mar = c(4, 4, 2, 2))
plot(cv.ridge)

# Evaluamos el error de prueba con bestlam_ridge
x.test = model.matrix(Apps ~ ., college.test)[, -1]
y.test = college.test$Apps
ridge.pred = predict(cv.ridge, s = bestlam_ridge, newx = x.test)
mean((ridge.pred - y.test)^2) # MSE = 2530947
rmse = sqrt(mean((ridge.pred - y.test)^2))
rmse # RMSE = 1590.9

# Evaluamos el error de prueba con bestlam2_ridge
ridge.pred2 = predict(cv.ridge, s = bestlam2_ridge, newx = x.test)
mean((ridge.pred2 - y.test)^2) # MSE = 3552978
rmse = sqrt(mean((ridge.pred2 - y.test)^2))
rmse # RMSE = 1884.9

##### Inciso d) #####
##### Ahora con regresion lasso
cv.lasso = cv.glmnet(x, y, alpha = 1, nfolds=5, lambda = val_lambda)

bestlam_lasso = cv.lasso$lambda.min
cat("Mejor lambda para lasso:", bestlam_lasso, "\n") # 86.85114

bestlam2_lasso = cv.lasso$lambda.1se
cat("Lambda 1se para lasso:", bestlam2_lasso, "\n") # 1098.541

plot(cv.lasso)

# Evaluamos el error de prueba con bestlam_lasso
lasso.pred = predict(cv.lasso, s = bestlam_lasso, newx = x.test)
mean((lasso.pred - y.test)^2) # MSE = 2395665
rmse = sqrt(mean((lasso.pred - y.test)^2))
rmse # RMSE = 1547.8


# Evaluamos el error de prueba con bestlam2_lasso
lasso.pred2 = predict(cv.lasso, s = bestlam2_lasso, newx = x.test)
mean((lasso.pred2 - y.test)^2) # MSE = 3438634



# Coeficientes lasso
lasso.coef = predict(cv.lasso, s = bestlam_lasso, type = "coefficients")
lasso.coef

#                        s1
# (Intercept) -3710.2528792
# PrivateYes      .        
# Top10perc      14.8625790
# Top25perc       0.6708990
# F.Undergrad     0.6768938
# P.Undergrad     .        
# Outstate        .        
# Room.Board      0.4678836
# Books           .        
# Personal       -0.0780206
# PhD             .        
# Terminal        .        
# S.F.Ratio      10.1380096
# perc.alumni    -0.5097409
# Expend          0.0612559
# Grad.Rate      18.3611036

#### Inciso e) #####
# Ahora ajustamos un modelo PCR con M elegido por validación cruzada
library("pls")

pcr.fit = pcr(Apps ~ ., data = college.train, scale = TRUE, validation = "CV")
summary(pcr.fit)

# 6 componentes explican el 80.83% de la varianza en X
# y un 61.73% de la varianza en Apps

# 9 componentes explican el 91.10% de la varianza en X
# y un 62.66% de la varianza en Apps

validationplot(pcr.fit,val.type="MSEP")
# El MSEP mas pequeño ocurre con M = 12 componentes
# Evaluamos el error de prueba con M = 12 componentes
pcr.pred = predict(pcr.fit, college.test, ncomp = 12)
mean((pcr.pred - y.test)^2) # MSE = 2389249
rmse = sqrt(mean((pcr.pred - y.test)^2))
rmse # RMSE = 1545.7

##### Inciso f) #####
# Ahora ajustamos un modelo PLS con M elegido por validación cruzada
pls.fit = plsr(Apps ~ ., data = college.train, scale = TRUE, validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")


# Evaluamos el error de prueba con M = 4 componentes

pls.pred = predict(pls.fit, college.test, ncomp = 4)
mean((pls.pred - y.test)^2) # MSE = 2457676
rmse = sqrt(mean((pls.pred - y.test)^2))
rmse # RMSE = 1567.7


##### Inciso g) #####

#### Con que precision se pueden predecir las aplicaciones?

















