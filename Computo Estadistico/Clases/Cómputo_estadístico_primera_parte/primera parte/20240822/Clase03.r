##
# Hosmer, D.W. & Lemeshow, S.(1989) Applied logistic regression. Wiley
# Edad y Coronaria (da˜no significativo en coronaria)
##

edad <- c(
20, 23, 24, 25, 25, 26, 26, 28, 28, 29, 30, 30, 30, 30, 30, 30, 32, 32, 33, 33,
34, 34, 34, 34, 34, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 39, 39, 40, 40, 41,
41, 42, 42, 42, 42, 43, 43, 43, 44, 44, 44, 44, 45, 45, 46, 46, 47, 47, 47, 48,
48, 48, 49, 49, 49, 50, 50, 51, 52, 52, 53, 53, 54, 55, 55, 55, 56, 56, 56, 57,
57, 57, 57, 57, 57, 58, 58, 58, 59, 59, 60, 60, 61, 62, 62, 63, 64, 64, 65, 69)

coro <- c(
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,
0,0,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,0,
0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1)

# Gráfica de los datos
edaj <- jitter(edad) # solo con fines de graficaci´on

plot(edaj, coro, xlab="Edad", ylab="Indicador CHD", ylim=c(-.1,1.1),
mgp=c(1.5,.5,0), cex.axis=.8, cex.lab=.8, cex.main=1, xlim=c(15,75), cex=.7,
main="Regresión lineal", pch=ifelse(coro==1,"1","0"))
rug(edaj)
out <- lm(coro ~ edad)
abline(reg=out$coef,lwd=2,col="blue")

plot(edaj, coro, xlab="Edad", ylab="Indicador CHD", ylim=c(-.1,1.1),
mgp=c(1.5,.5,0), cex.axis=.8, cex.lab=.8, cex.main=1, xlim=c(15,75), cex=.7,
main="Regresión Logística", pch=ifelse(coro==1,"1","0"))
rug(edaj)

# Resolviendo ecuaciones de verosimilitud
y <- coro
n <- length(y)
X <- cbind(rep(1,n),edad)
b <- c(-10,.2) # valores iniciales
# Las 4 l´ýneas anteriores son espec´ýficas para los datos de coronaria
tolm <- 1e-6 # tolerancia (norma minima de delta)
iterm <- 100 # numero maximo de iteraciones
tolera <- 1 # inicializar tolera
itera <- 0 # inicializar itera
histo <- b # inicializar historial de iteraciones
while( (tolera>tolm)&(itera<iterm) ){
p <- 1/( 1+exp( -as.vector(X%*%b) ) )
W <- p*(1-p)
delta <- as.vector( solve(t(X*W)%*%X, t(X)%*%(y-p)) )
b <- b + delta
tolera <- sqrt( sum(delta*delta) )
histo <- rbind(histo,b)
itera <- itera + 1 }
# histo -10.000000 0.20000000
# b -1.497206 0.03767488
# b -4.380358 0.09253679
# b -5.221685 0.10918224
# b -5.308597 0.11090419
# b -5.309453 0.11092114
# b -5.309453 0.11092114
# Agregamos curva log´ýstica a la gr´afica original
xx <- seq(15,75,length=200)
X <- cbind(rep(1,n),xx)
p <- 1/( 1+exp( -as.vector(X%*%b) ) )
lines(xx,p,lwd=2,col="blue")
grid(lwd=2)

##
# Errores Estándar
##

y <- coro
n <- length(y)
X <- cbind(rep(1,n),edad)
# Las l´ýneas anteriores son espec´ýficas para los datos CHD
p <- 1/( 1+exp( -as.vector(X%*%b) ) )
W <- p*(1-p)
V <- solve( t(X*W)%*%X )
es <- sqrt( diag(V) )
# estimadores y sus desviaciones est´andar
# b es
# -5.3094534 1.13365464
# edad 0.1109211 0.02405984
# Usando glm
out <- glm(y ~ edad, family=binomial)
summary(out)

# Deviance Residuals:
# Min 1Q Median 3Q Max
# -1.9718 -0.8456 -0.4576 0.8253 2.2859
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)
# (Intercept) -5.30945 1.13365 -4.683 2.82e-06 ***
# edad 0.11092 0.02406 4.610 4.02e-06 ***
# Null deviance: 136.66 on 99 degrees of freedom
# Residual deviance: 107.35 on 98 degrees of freedom
# AIC: 111.35
# Number of Fisher Scoring iterations: 4

################################################################################

# Consideremos un estudio sobre pesos de recién nacidos. Se
# tienen 188 registros de nacimientos de los cuales 58 fueron
# bebés de peso bajo (< 2.5 kgm).

lowbwt <- read.table( "lowbwt.txt", header=T, sep="|" )
lowbwt[1:4,]

catftv <- data.frame(ftv=sort(unique(lowbwt[,"ftv"])),NumV=c(0,1,2))
lowbwt[,"ftv"] <- catftv[match(lowbwt[,"ftv"],catftv[,1]),2]

attach(lowbwt)

table(race)
table(ftv)

low <- ifelse(bwt<2500,1,0)
r1  <- ifelse(race=="Black",1,0)
r2  <- ifelse(race=="Other",1,0)
out <- glm(low ~ age+lwt+r1+r2+ftv, family=binomial)
summary(out)

# Resolviendo ecuaciones de verosimilitud a "pie"
y <- low
n <- length(y)
X <- cbind(rep(1,n),age,lwt,r1,r2,ftv)
b <- c(1,0,0,1,.3,0) # valores iniciales
tolm <- 1e-6 # tolerancia (norma minima de delta)
iterm <- 100 # numero maximo de iteraciones
tolera <- 1 # inicializar tolera
itera <- 0 # inicializar itera
histo <- b # inicializar historial de iteraciones
while( (tolera>tolm)&(itera<iterm) ){
p <- 1/( 1+exp( -as.vector(X%*%b) ) )
W <- p*(1-p)
delta <- as.vector( solve(t(X*W)%*%X, t(X)%*%(y-p)) )
b <- b + delta # al final, b = estimadores Max.V.
tolera <- sqrt( sum(delta*delta) )
histo <- rbind(histo,b)
itera <- itera + 1 }

histo

# C´alculo de errores est´andar
n <- length(y)
X <- cbind(rep(1,n),age,lwt,r1,r2,ftv)
p <- 1/( 1+exp( -as.vector(X%*%b) ) )
W <- p*(1-p)
V <- solve( t(X*W)%*%X )
es <- sqrt( diag(V) )
cbind(b,es)


# Prueba global
# -2loglik del modelo (residual deviance)
aa <- -2*(sum(y*log(p) + (1-y)*log(1-p))) # 222.3098
# -2loglik del modelo nulo (null deviance)
bb <- -2*(sum(y)*log(mean(y))+(n-sum(y))*log(1-mean(y))) # 234.672
# -2 log(Cociente de verosimilitudes)
G <- bb-aa # 12.36
pval <- 1-pchisq(G,df=5) # 0.030 => rech Ho

z <- b[6]/es[6] # -0.589856
2*(1-pnorm(abs(z))) # 0.5552872


# modelo completo vs reducido
mcomp <- glm(low ~ age+lwt+r1+r2+ftv, family=binomial)
mred <- glm(low ~ lwt+r1+r2, family=binomial)
G <- mred$deviance - mcomp$deviance # 0.949
pval <- 1-pchisq(G,df=2) # 0.622
# no rechazamos el modelo nulo (i.e. el modelo reducido)

mred <- glm(low ~ lwt+r1+r2, family=binomial)
summary(mred)

br <- mred$coeff
xx <- seq(80,250,length=200)
X <- cbind(rep(1,200),xx)
p <- 1/( 1+exp( -as.vector(X%*%br[1:2]) ) )
p1 <- 1/( 1+exp( -as.vector(X%*%(br[1:2]+c(br[3],0))) ) )
p2 <- 1/( 1+exp( -as.vector(X%*%(br[1:2]+c(br[4],0))) ) )
plot(xx,p,xlab="Peso de Madre (lbs)",
ylab="Probabilidad de Peso Bajo de Bebé",
ylim=c(-.1,.8), mgp=c(1.5,.5,0),cex.axis=.8,cex.lab=.8,
cex.main=1,xlim=c(80,250),cex=.7,lwd=2,col="blue",type="l",
main="Peso de recién nacido vs peso de madre")
lines(xx,p1,lwd=2,col="red")
lines(xx,p2,lwd=2,col="green")
rug(jitter(lwt))
legend(200,.8,legend=c("Blanca","Negra","Otra"),lwd=2,
col=c("blue","red","green"))

