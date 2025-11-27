## Gustavo Hernandez Angeles
## Problema 1. Examen 2 de Computo Estadistico

### Utilizaremos los datos de Salary.csv este archivo contiene datos de salarios de empleados en una empresa como variable dependiente 
### y años de experiencia como variable independiente.

# Cargar las librerias necesarias
library(ggplot2)


# Cargar los datos
data <- read.csv("Salary.csv")

# Graficar los datos
ggplot(data, aes(x = YearsExperience, y = Salary)) +
    geom_point() +
    labs(title = "Datos Completos",
         x = "Años de Experiencia",
         y = "Salario") +
    theme_minimal()


### Inciso a)
### Escribe un programa en R que elimine aproximadamente la mitad de los 
### valores de x. Diseña este mecanismo de no respuesta para que sea aleatorio
### pero no completamente aleatorio, es decir, la probabilidad de que x sea faltante 
### debe depender de y. Llama a este nuevo conjunto de datos (con faltantes en x)
### los "datos disponibles".

set.seed(123)  # Para reproducibilidad
n <- nrow(data)
prob_missing <- plogis((data$Salary - mean(data$Salary)) / sd(data$Salary))  # Probabilidad de que x sea faltante depende de y
missing_indices <- rbinom(n, 1, prob_missing) == 1
data_available <- data
data_available$YearsExperience[missing_indices] <- NA

ggplot(data_available, aes(x = YearsExperience, y = Salary)) +
    geom_point() +
    labs(title = "Datos Disponibles",
         x = "Años de Experiencia",
         y = "Salario") +
    theme_minimal()

### Inciso b)
### Realiza la regresion de x sobre y utilizando solo los datos para los que se
### observan ambas variables (los "datos disponibles") y demuestra que es 
### coherente con la regresion utilizando los datos completos.

model_available <- lm(YearsExperience ~ Salary, data = data_available, na.action = na.omit)
summary(model_available)

model_complete <- lm(YearsExperience ~ Salary, data = data)
summary(model_complete)

### Inciso c)
### Realiza la regresion de y sobre x utilizando los datos disponibles y demuestra
### que no es consistente con la regresion utilizando los datos completos.

model_y_on_x_available <- lm(Salary ~ YearsExperience, data = data_available, na.action = na.omit)
summary(model_y_on_x_available)

model_y_on_x_complete <- lm(Salary ~ YearsExperience, data = data)
summary(model_y_on_x_complete)
