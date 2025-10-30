##práctica sobre validación cruzada
# se trabajara con los datos de los coches

######  Enfoque del conjunto de validación ######

library(ISLR) # contiene los conjuntos de datos analizados en el libro Introduction to Statistical learning with R 

set.seed(1) # usamos esta función para fijar una semilla para los generadores de numeros aleaorios en R

train <- sample(392,196) #esta función selecciona aleatoriamente un subconjunto de 196 numeros indicadores y a
train
#partir de estos se selecciona el conjunto de entrenamiento


lm.fit<-lm(mpg~horsepower, data=Auto,subset=train) #se ajusta una regresion lineal usando los datos de entrenamiento

attach(Auto) #adjunta las variables del conjunto de datos que se evaluaran, dando su nombre
mean((mpg -predict (lm.fit,Auto))[-train ]^2) #calcula el MSE con las observaciones del conjunto de validación (error de prueba)

lm.fit2<-lm(mpg~poly(horsepower,2) ,data=Auto,subset =train) #se ajusta una regresion con un polinomio de grado 2
mean((mpg -predict (lm.fit2,Auto))[-train ]^2) #calcula el MSE de las observaciones en el conjunto de validación (error de prueba)
lm.fit3<-lm(mpg~poly(horsepower,3) ,data=Auto ,subset =train)#se ajusta una regresion con un polinomio de grado 3
mean((mpg -predict (lm.fit3,Auto))[-train ]^2)

lm.fit4<-lm(mpg~poly(horsepower,4) ,data=Auto ,subset =train)#se ajusta una regresion con un polinomio de grado 4
mean((mpg -predict (lm.fit4,Auto))[-train ]^2)



#Los errores de prueba son 26.14, 19.82, 19.78 y 19.99 respectivamente,

#eligiendo conjunto de entrenamiento diferente

set.seed(2) #fijamos otra semilla para generar un conjunto de datos de entrenamiento distinto

train <- sample(392,196) 
lm.fit<-lm(mpg~horsepower,data=Auto,subset=train) #se ajusta una regresion lineal usando los datos de entrenamiento
mean((mpg -predict (lm.fit,Auto))[-train ]^2) #calcula el MSE de las observaciones en el conjunto de validación (error de prueba)

lm.fit2<-lm(mpg~poly(horsepower,2),data=Auto,subset =train) #se ajusta una regresion con un polinomio de grado 2
mean((mpg -predict (lm.fit2,Auto))[-train ]^2) #calcula el MSE de las observaciones en el conjunto de validación (error de prueba)
lm.fit3<-lm(mpg~poly(horsepower,3) ,data=Auto ,subset =train)#se ajusta una regresion con un polinomio de grado 3
mean((mpg -predict (lm.fit3,Auto))[-train ]^2)

lm.fit4<-lm(mpg~poly(horsepower,4) ,data=Auto ,subset =train)#se ajusta una regresion con un polinomio de grado 4
mean((mpg -predict (lm.fit4,Auto))[-train ]^2)


#conclusión: un modelo cuadratico predice mejor el rendimiento, que un modelo que implica
#una funcion lineal y existe poca evidencia en favor de un modelo que usa una funcion
#cubica o de grado 4 de los caballos de fuerza.



######  validacion cruzada dejando uno fuera (LOOCV) ######

library (boot) # contiene  la funcion de validacion cruzada
glm.fit<-glm(mpg~horsepower,data=Auto) #aqui aplicamos un modelo lineal generalizado, porque
#la validacion cruzada solo se calcula en terminos de modelos generalizados
cv.err<-cv.glm(Auto,glm.fit,K=nrow(Auto)) #aquí se calcula el error de validacion
#cruzada (error de prueba) para la LOOCV
#
cv.err$delta[1] #aqui nos da dos valores del error de validacion cruzada, debido a que cuando
#los datos originales se dividen en k pliegues muchas veces los tamaños de los pliegues
#no son iguales, entonces en ese caso se realiza un ajuste en el error de validación cruzada que es el segundo
#valor del vector

#en este ciclo obtenemos el error de LOOCV para cada polinomio ajustado(de grado 1 hasta 10)
cv.error<-rep (0,10) #se inicializa el vector en el que se guardaran los errores de LOOCV
#para cada polinomio ajustado, de grado 1 hasta 10
  for (i in 1:10){
  glm.fit<-glm(mpg~poly(horsepower,i),data=Auto)
  cv.error[i]<-cv.glm(Auto,glm.fit)$delta [1]
   }
cv.error
#24.23151 19.24821 19.33498 19.42443 19.03321 18.97864 18.83305 18.96115 19.06863 19.49093
x<-1:10
plot(x,cv.error,type="o",xlab="Grado del polinomio",ylab="Error de prueba")

#conclusion: el polinomio de grado 2 presenta un error de prueba pequeño y similar a los obtenidos
#en los polinomios de orden mayor


######  validacion cruzada k-fold ######
#considerando k=10
set.seed (17)

#en este ciclo obtenemos el error de CV k-fold para cada polinomio ajustado (de grado 1 hasta 10)
cv.error.10<- rep (0,10) #se inicializa el vector en el que se guardaran los errores de CV k-fold para cada polinomio ajustado, de grado 1 hasta 10 
for (i in 1:10) {
  glm.fit<-glm(mpg~poly(horsepower,i),data=Auto)
  cv.error.10[i]<-cv.glm (Auto,glm.fit,K=10)$delta [1]
   }
cv.error.10

#24.20520 19.18924 19.30662 19.33799 18.87911 19.02103 18.89609 19.71201 18.95140 19.50196
x<-1:10
plot(x,cv.error.10,type="o",xlab="Grado del polinomio",ylab="Error de prueba")


#nota: el tiempo de ejecucion  del cv k-fold es menor al LOOCV porque k<n

#conclusion: existe poca evidencia de que usar un polinomio cubico o de orden
#mayor proporciona un error de prueba mas bajo, que simplemente usar un ajuste cuadratico

