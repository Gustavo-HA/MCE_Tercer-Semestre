## Gustavo Hernandez Angeles
## Problema 2. Examen 2 de Computo Estadistico

# Instalar paquetes necesarios si no estan instalados
if (!require("readxl")) install.packages("readxl")
if (!require("naniar")) install.packages("naniar")
if (!require("mice")) install.packages("mice")

library(readxl)
library(naniar) # Para el test de Little
library(mice)   # Para imputacion multiple

### Utilizaremos los datos Cancer_data.xls

data <- read_excel("Cancer_data.xls")

# Los datos faltantes son los que tienen por valor -9
data[data == -9] <- NA

# Tanto las columnas SexP tienen
# 1 para masculino y 2 para femenino, lo convertimos a binario
data$SexP <- ifelse(data$SexP == 1, 0, 1)

# Seleccionamos las variables de interes para el analisis
# Dependiente: Totbpt
# Predictoras: SexP, AnxtP, DeptP, AnxtS, DeptS
vars_interes <- c("Totbpt", "SexP", "AnxtP", "DeptP", "AnxtS", "DeptS")
data_subset <- data[, vars_interes]

### Inciso (a)
### Considerando solo los casos completos construye un modelo de regresion lineal múltiple 
### para predecir la variable dependiente Totbpt con las 5 variables predictoras mencionadas.


# lm omite automaticamente los NA (casos completos)
model_cc <- lm(Totbpt ~ SexP + AnxtP + DeptP + AnxtS + DeptS, data = data_subset)
summary(model_cc)

### Inciso (b)
### Aplica el test de Little para probar si los datos faltantes siguen un patrón 
### completamente aleatorio (MCAR).

# H0: Los datos son MCAR (Missing Completely At Random)
# Si p-value > 0.05, no rechazamos H0 y asumimos MCAR.
mcar_test_result <- mcar_test(data_subset)
print(mcar_test_result)

### Inciso (c)
### Aplica el enfoque de máxima verosimilitud (con algoritmo EM) o el enfoque de 
### imputación múltiple para estimar los datos faltantes en todas las variables.

# Convertimos la variable binaria a factor para usar regresion logistica en la imputacion
data_subset$SexP <- as.factor(data_subset$SexP)

# Configuramos los metodos de imputacion:
# 'logreg' para la variable binaria (SexP)
# 'pmm' (Predictive Mean Matching) para las variables continuas
ini <- mice(data_subset, maxit = 0)
meth <- ini$method
meth["SexP"] <- "logreg"
meth[names(meth) != "SexP"] <- "pmm"

set.seed(123)
imputed_data <- mice(data_subset, m = 5, method = meth, printFlag = FALSE)
cat("Resumen de la imputación:\n")
print(imputed_data)

### Inciso (d)
### Con los datos completos obtenidos en (c), construye un modelo de regresión lineal 
### y compara los resultados con los obtenidos en (a).

# Ajustamos el modelo a cada uno de los datasets imputados y combinamos (pool) los resultados
model_mi <- with(imputed_data, lm(Totbpt ~ SexP + AnxtP + DeptP + AnxtS + DeptS))
pooled_results <- pool(model_mi)
summary(pooled_results)

cat("\n--- Comparación de Resultados ---\n")
cat("Coeficientes Casos Completos:\n")
print(coef(model_cc))
cat("\nCoeficientes Imputación Múltiple (Pooled):\n")
print(summary(pooled_results)[, c("term", "estimate", "std.error", "p.value")])



