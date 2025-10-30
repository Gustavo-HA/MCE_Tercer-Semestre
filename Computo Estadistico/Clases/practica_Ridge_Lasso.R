##práctica sobre Regresión Rdige y Lasso
# se trabajara con el conjunto de datos "Hitters", que representan el salario de una muestra de peloteros de grandes ligas y algunas estadisticas de bateo
#de cada uno de ellos en la temporada anterior. Se busca  predecir el salario  de los jugadores en bases a sus estadisticas.
#(Hrs, hits, RBI,walk, etc)

######  Regresion Ridge ######
#primero usaremos Ridge para predecir los salarios

library(ISLR) # contiene los conjuntos de datos analizados en el libro Introduction to Statistical learning with R 
library(glmnet) #contiene los procedimientos de regresion Ridge, lasso y otros metodos de contracción

Hitters
dim(Hitters)

#prueba
College
dim(College)


x<-model.matrix (Salary~.,Hitters )[,-1] # funcion que crea una matriz x, donde las columnas representan las variables predictoras
# tambien transforma automaticamente cualquier varible cualitativa a variables dummy, esto es importante porque glmnet()
#considera unicamente datos de entrada cuantitativos

y<-Hitters$Salary # se seleccionan los salarios de los jugadores, que serán la respuesta Y 
grid<-10^seq(10,-2,length =100) # la funcion glmnet() selecciona de manera automatica un rango de valores para lamda
#pero aqui fijamos el rango de valores de lamda, de lamda=10^10 a lamda=10^-2,obteniendo 100 valores

 
ridge.mod<-glmnet(x,y,alpha =0,lambda =grid) #funcion que ajusta los modelos con los
#metodos de contracción, si alpha=0 ajusta una regresion ridge, 
#si alpha=1 ajusta un modelo lasso. Usaremos primero regresion ridge.
#Nota: la funcion glmnet estandariza previamente las variables para que esten en la misma escala

coefridge_rango<-coef(ridge.mod) # contiene los estimadores de los betas obtenidos
#para cada valor de lamda, es una matriz de 20 x 100
dim(coefridge_rango)
coefridge_rango[,1:50] # se muestran los coeficientes estimados para los primeros 50 valores de lamda (los mas grandes) 
coefridge_rango[,51:100] # se muestran los coeficientes estimados para los ultimos 50 valores de lamda (los mas pequeños)

lamda50<-ridge.mod$lambda [50] #muestra el valor 50 de lamda
lamda50
coef.lamda50<-coef(ridge.mod)[,50] #muestra los coeficientes estimados beta para ese valor de lamda
coef.lamda50
l2.coeflamdda50<-sqrt(sum(coef(ridge.mod)[-1,50]^2)) # se calcula la norma l2 del vector
#de estimaciones de las betas para ese valor de lamda
l2.coeflamdda50

ridge.mod$lambda [100] #muestra el  valor 100 de lamda (el lamda mas pequeño)
coef(ridge.mod)[,100] #muestra los coeficientes estimados para ese valor de lamda
#al ser el lamda mas pequeño,  se espera que los coeficientes estimados sean
#diferentes a cero y similares a los obtenidos con minimos cuadrados

# se calcula la norma l2 del vector de estimaciones de las betas para ese valor de lamda

l2.coeflamdda100<-sqrt(sum(coef(ridge.mod)[-1,100]^2)) 
l2.coeflamdda100 # es mas grande que la norma l2 de los coeficiente con el valor 50 de
#lamda


#vamos a seleccionar el lamda que nos proporcione el mejor modelo. Para esto usaremos
#el procedimiento de validacion con un solo conjunto de validación

set.seed(1) #fijamos la semilla para reproducir los resultados
train<-sample (1:nrow(x), nrow(x)/2) # se selecciona aleatoriamente la mitad de obs para el
#conjunto de entrenamiento
test<-(-train) #la otra mitad sera para probar el modelo (conjunto de validacion)

y.test<-y[test] #se elige los salarios de los jugadores para los datos de prueba (var. respuesta)

# se ajusta la regresion ridge, con los datos de entrenamiento, para el rango de 
#valores de lamda  dado y con un umbral  muy pequeño para la convergencia del
#algoritmo utilizado

ridge.mod <-glmnet (x[train ,],y[train],alpha =0, lambda =grid,thresh =1e-12) 

#inicia ciclo para calcular el error de validacion para cada uno de los valores de lamda
#dados en el grid

cv.error1<- rep (0,100) #se inicializa el vector en el que se guardaran los errores
#de prueba (errores de validacion), considerando un conjunto de validacion


for (i in 1:100){


#se predicen los valores del conjunto de prueba considerando el modelo ajustado con cada
#valor de lamda, usando los valores x del conjunto de prueba
  ridge.pred<-predict(ridge.mod,s=grid[i],newx=x[test,]) 

#se calcula el error de prueba MSE para cada valor de lamda
  cv.error1[i]<-mean((ridge.pred-y.test)^2) 

}

cv.error1 #vector con los errores de prueba para cada valor de lamda
plot(grid,cv.error1) #no es muy claro el gráfico por las escalas de los ejes
#print(cv.error1)
#print(cv.error2)

minMSE1<-min(cv.error1) #valor minimo del error de prueba
minMSE1
lamda.opt<-grid[which.min(cv.error1)] # valor del lamda optimo (con el que se obtuvo el 
#valor minimo del error )
lamda.opt


# se calcula el error de prueba para el modelo con el lamda mas grande (el modelo nulo) 
ridge.pred<-predict(ridge.mod,s=1e10,newx=x[test,])
MSE.nulo<-mean((ridge.pred-y.test)^2) # se calcula el error de prueba MSE
MSE.nulo
#esto quiere decir que un modelo de regresion ridge con el lamda adecuado
#tiene un error de prueba mucho mas bajo que un modelo casi nulo


#vamos a revisar cual es el beneficio de realizar una regresion ridge con
#lamda=231.013 (optimo) en lugar de  usar minimos cuadrados

# se calcula el error de prueba para el modelo con el lamda mas pequeño (ajuste
#por minimos cuadrados)

ridge.pred<-predict(ridge.mod,s=.01,newx=x[test,])
MSE.mincua<-mean((ridge.pred-y.test)^2)
MSE.mincua
#esto quiere decir que un modelo de regresion ridge con el lamda adecuado
#tiene un error de prueba mucho mas bajo que el ajustado  por minimos cuadrados

#lm(y~x,subset=train)
#predict(ridge.mod,s=0,exact=T,type="coefficients") [1:20,]

#vamos a seleccionar el lamda que nos proporcione el mejor modelo, pero ahora
#usaremos el procedimiento de VC -kfold
set.seed(1)

#funcion que realiza la validacion cruzada k-fold, por default usa k=10, 
#usando el mismo rango de valores para lamda

cv.out<-cv.glmnet(x,y,alpha =0,lambda =grid)
#cv.out<-cv.glmnet(x[train,],y[train],alpha =0,lambda =grid) 

plot(cv.out) #grafica MSE de prueba para cada valor de lamda
bestlam<-cv.out$lambda.min #elige el valor de lamda que tiene el MSE de prueba mas pequeño
bestlam

#el cual nos da el mismo valor optimo de lamda que el obtenido
#con la validacion de un solo subconjunto

#Finalmente, reajustamos el modelo de regresión Ridge en el conjunto de datos completo,
#utilizando el valor de lamda elegido por la validación cruzada, y examinamos
#las estimaciones de los coeficientes.

out<-glmnet(x,y,alpha=0,lambda =grid) #ajustamos el modelo de regresión Ridge con los datos completos
predict(out,type="coefficients",s=bestlam)[1:20,]
 # como se esperaba, ninguno de los coeficientes son cero, la regresion Ridge no
#realiza seleccion de variables


######  Regresion Lasso ######
##se realiza de manera similar a Ridge, solo que alpha=1
lasso.mod<-glmnet(x,y,alpha =1,lambda=grid)
plot(lasso.mod)

#vamos a seleccionar el lamda que nos proporcione el mejor modelo. 
#Para esto ahora usaremos el procedimiento de VC -kfold

set.seed(1)
#funcion que realiza la validacion cruzada k-fold, por default usa k=10
cv.out<-cv.glmnet(x,y,alpha=1,lambda =grid)
#cv.out<-cv.glmnet(x[train,],y[train],alpha =1,lambda =grid) 

plot(cv.out) #grafica los MSE para cada valor de lamda
bestlam<-cv.out$lambda.min  #elige el valor de lamda que tiene el MSE mas pequeño
bestlam


#Finalmente, reajustamos el modelo de regresión lasso en el conjunto de datos completo, 
#utilizando el valor de lamda elegido por la validación cruzada, y examinamos las
#estimaciones de los coeficientes.

out<-glmnet(x,y,alpha=1,lambda =grid)
lasso.coef<-predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef

#comentarios finales: 
# lasso tiene una ventaja sustancial sobre la regresión Ridge: que las estimaciones de
#algunos coeficiente resultantes son cero. Aquí vemos que 6 de los 19  coeficientes
#estimados son exactamente cero. Así, el modelo lasso con el lamda elegido por
#validación cruzada contiene sólo 13 variables predictoras.
