
dat <- read.csv("Datos_Polio.csv", header=TRUE)

Mes   <- colnames(dat)
casos <- NULL
for(i in 1:nrow(dat)){# i <- 1
  aux <- data.frame(Year=dat[i,1],Count=as.numeric(dat[i,2:13]),Month=1:12+(i-1)*12,Mes=1:12)
  casos <- rbind(casos,aux)
}

attach(casos)

plot(Month,Count,pch=16,xlab="Mes",ylab="Número de casos",
main="Casos de Polio")

plot(Month,Count,type="l",xlab="Mes",ylab="Número de casos",
main="Casos de Polio", lwd=2)

aa <- matrix(casos[,2],12,14)

med <- apply(aa,2,mean)
vari <- apply(aa,2,var)

plot(med,vari,pch=16,xlab="Promedio anual de casos de Polio",
ylab="Varianza",main="Relación Media vs Varianza")
abline(a=0,b=1,lwd=2,col="blue")
grid()

plot(1:14,med,pch=16,xlab="Promedio anual de casos de Polio",
ylab="media anual",main="Casos de Polio por Año")
grid()

###

out <- glm(Count ~ Month, family=poisson)
summary(out)

#Call:
#glm(formula = Count ~ Month, family = poisson)
#
#Deviance Residuals: 
#    Min       1Q   Median       3Q      Max  
#-1.9305  -1.4734  -0.4244   0.5052   5.9791  
#
#Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
#(Intercept)  0.626639   0.123641   5.068 4.02e-07 ***
#Month       -0.004263   0.001395  -3.055  0.00225 ** 
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#(Dispersion parameter for poisson family taken to be 1)
#
#    Null deviance: 343.00  on 167  degrees of freedom
#Residual deviance: 333.55  on 166  degrees of freedom
#AIC: 594.59
#
#Number of Fisher Scoring iterations: 5
#

out2 <- glm(Count ~ Month, family=quasipoisson)
summary(out2)

#Call:
#glm(formula = Count ~ Month, family = quasipoisson)
#
#Deviance Residuals: 
#    Min       1Q   Median       3Q      Max  
#-1.9305  -1.4734  -0.4244   0.5052   5.9791  
#
#Coefficients:
#             Estimate Std. Error t value Pr(>|t|)   
#(Intercept)  0.626639   0.194788   3.217  0.00156 **
#Month       -0.004263   0.002198  -1.939  0.05415 . 
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#(Dispersion parameter for quasipoisson family taken to be 2.481976)
#
#    Null deviance: 343.00  on 167  degrees of freedom
#Residual deviance: 333.55  on 166  degrees of freedom
#AIC: NA
#
#Number of Fisher Scoring iterations: 5
#

out = glm(Count ~ Month, family=poisson)
aa = summary(out)
beta0 = aa$coefficients[1,1]
beta1 = aa$coefficients[2,1]
lamb = exp(beta0+beta1*Month)
# logverosimilitud
loglik1 = sum( -lamb + Count*log(lamb) - lfactorial(Count) )
# logverosimilitud saturada
likS = prod( (exp(-Count)*(Count^Count)) / factorial(Count) )
loglikS = log(likS)
# aproximadamente igual a:
loglikSS = sum( -Count + Count*log(ifelse(Count==0,Count+.1,Count)) - lfactorial(Count) )

# logverosimilitud nula
med = mean(Count)
loglikN = sum( -med + Count*log(med) - lfactorial(Count) )
devi = -2*(loglik1 - loglikS) # = 333.5466 "Residual deviance"
aic = -2*loglik1 + 2*2 # = 594.5895 "AIC"
deviN = -2*(loglikN - loglikS) # = 343.0004 "Null deviance"
# residuales de devianza
dd = -2*(-lamb + Count*log(lamb) + Count - Count*log(ifelse(Count==0,Count+.1,Count)) )
resd = sign(Count-lamb)*sqrt(dd)
summary(resd) # = "Deviance Residuals"
# Min. 1st Qu. Median Mean 3rd Qu. Max.
# -1.9305 -1.4734 -0.4244 -0.3027 0.5052 5.9791

