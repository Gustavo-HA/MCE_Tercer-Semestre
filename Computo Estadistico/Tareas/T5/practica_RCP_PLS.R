##práctica sobre Regresion por componentes principales
# se trabajara con los datos que relacionan  los salarios anuales de los jugadores de beisboly sus estadisticas en la temporada anterior.
#Hitters data. El objetivo es predecir los salarios de los jugadores


library(ISLR) # contiene los conjuntos de datos analizados en el libro Introduction to Statistical learning with R 
library(glmnet) #contiene los procedimientos de regresion Ridge, lasso y otros metodos de contracción
library(pls)  #contiene  las funciones que realizann PCR y PLS

set.seed(2) # usamos esta función para fijar una semilla para los generadores de numeros aleaorios en R
pcr.fit<-pcr(Salary~.,data=Hitters,scale=TRUE,validation="CV")  #es similar a lm(). Al fijar scale= T, se estandariza cada predictor, Fijando 
#validation= CV, calcula los errores de VC 10-fold, para cada M, el numero de componentes pricipales usado.

summary (pcr.fit) #
# el metodo para obtener los componentes es por descomposicion en valores singulares: svdpc
# Los valores de los errores de VC para cada numero de componentes M: 0(solo intercepto), 1,,..,p
#Nota:pcr() reporta la raiz del error cuadratico medio (RMSEP), asi por ejemplo el RMSEP en M=4 es 352.8, entonces el MSE=124,468

validationplot(pcr.fit,val.type="MSEP") # grafica los valores de los errores de VC para cada M, el número de componentes principales
 # al poner el tipo MSEP, grafica los cuadrados de RMSEP (los errores cuadraticos medios)
#la linea punteada en rojo son los MSEP ajustados
# el error mas pequeño ocurre en M=16, con lo cual no ganamos mucho en reducir la dimension porque p=19, y por tanto
#equivale a realizar  mínimos cuadrados, porque casi todos los componentes se usan. 

#Sin embargo, a partir de la gráfica también vemos que el error de validación cruzada es aproximadamente
#igual cuando sólo se incluye un componente en el modelo. Esto sugiere que un modelo que
#utiliza sólo un pequeño número de componentes podría ser suficiente.

#En el resumen del procedimiento también se proporciona el porcentaje de varianza explicada en los predictores y en la
#respuesta, para los diferentes números de componentes. 

#Podemos interpretar  estos porcentajes como la cantidad de información sobre los predictores o la respuesta
#que se captura utilizando M componentes principales. 
#Por ejemplo, el ajuste con M = 1 sólo captura 38.31 de toda la varianza, o información, en los predictores. Por el
#contrario, usando M = 6 capturamos el 88,63% de toda la varianza. Si tuviéramos que usar todos los componentes M=p=19,
#esto aumentaría al 100%.


#Nota: cuando construimos los componentes, no los relacionamos con la variable respuesta Y, es decir Y no supervisa la obtencion de
#los componentes, por tanto su relacion no tiene por que ser muy alta, o en otras palabras el % de variacion de la respuesta Y explicada
#por los componentes no tiene que ser muy alta
#De hecho, se observa que la varianza explicada en los salarios(respuesta)  por todos los componentes principales es menor a 55.

#Ahora realizamos la PCR sobre los datos de entrenamiento y evaluamos su desempeño en el conjunto de pruebas.

set.seed (1)
pcr.fit<-pcr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation ="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")


#Ahora observamos que el MSEP mas pequeño se presenta cuando se usan  M=7 componentes principales

#se calcula el MSE con los datos de prueba usando el modelo con 7 componentes
pcr.pred<-predict(pcr.fit,x[test,],ncomp =7) # esta funcion predice los valores de prueba de acuerdo al modelo ajustado con  7 componentes
mean((pcr.pred-y.test)^2) # se calcula el error de prueba


#Este error de prueba es competitivo con los resultados obtenidos usando la regresión Ridge y lasso. 
#Sin embargo, como resultado de la implementación de la PCR, el modelo final es más difícil de interpretar
#porque no realiza ningún tipo de selección de variables, ni siquiera produce directamente estimaciones de coeficientes


#Por último, se ajusta la PCR con el conjunto de datos completos, utilizando M=7 componentes
#identificados por la validación cruzada.
pcr.fit<-pcr(y~x,scale=TRUE,ncomp=7)
summary (pcr.fit)

#asi, el ajuste con M = 4 explica el casi el 80% de toda la varianza en los predictores.
#Mientras que con M=4, se explica unicamente el 43% de la varianza en los salarios (respuesta).

##Regresion por mínimos cuadrados parciales ####################################

# la implementación de mínimos parciales (PLS) se realiza mediante la función plsr (), que también está
#en la libreria de pls. La sintaxis es similar a la de la función pcr ()

#se utilizaran los mismos datos Hitters

#Realizamos la regresión PLS sobre los datos de entrenamiento y evaluamos su desempeño
#en el conjunto de prueba.

set.seed (1)

pls.fit<-plsr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation ="CV")
summary(pls.fit)
validationplot(pls.fit,val.type="MSEP")

#El error de validación cruzada mas pequeño ocurre cuando sólo se utilizan M=2 direcciones de mínimos
#cuadrados parciales. 

#Ahora evaluamos el error de prueba (MSE) correspondiente  a M=2.
pls.pred<-predict(pls.fit,x[test,],ncomp =2)
mean((pls.pred-y.test)^2)

#El error de prueba (MSE) es comparable al obtenido usando regresion Ridge,  lasso y
#PCR, aunqe ligeramente mas grande

#Finalmente, se realiza PLS considerando los datos completos, usando M = 2, el número de componentes identificados por la validación cruzada

pls.fit<-plsr(Salary~.,data=Hitters,scale=TRUE,ncomp=2)
summary (pls.fit)

#Observe que el porcentaje de varianza en Salario explicada por el ajuste de PLS de dos componentes
#es 46,40%, es casi tanto como el porcentaje explicado usando el ajuste final de siete componentes de PCR, 46,69%.
#Esto se debe a que la PCR solo intenta maximizar la cantidad de varianza explicada en los predictores, mientras que
#PLS busca las direcciones que explican la varianza tanto en los predictores como en la respuesta.
