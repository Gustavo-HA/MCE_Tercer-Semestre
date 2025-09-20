y <- c(12,14,33,50,67,74,123,141,165,204,253,246,240)
n <- length(y)
x <- 1:n

par(mfrow=c(1,2))
plot(x+1980,y,xlab="Year",ylab="New AIDS cases",
ylim=c(0,280),pch=16, main="Y vs X")
plot(x+1980,log(y),xlab="Year",ylab="Log(New AIDS cases)",
ylim=c(2,6), main="log(Y) vs X",pch=16)

##
# Metodo Directo de Maxizar la LogVerosimilitud
##

Y <- y
X <- cbind(rep(1,n),x,x*x) # modelo cuadratico

FunlogV <- function(Bet){
  Lami <- exp(X%*%Bet)
  logV <- sum( Y*log(Lami) - Lami )
  return(-logV)
}

Bet <- c(2,1,0)# valores iniciales de par�metros
# FunlogV(Bet)

nlminb(Bet,FunlogV)

##
# Metodo de Newton
##

b      <- c(2,1,0) # valores iniciales de par�ametros
tolm   <- 1e-6 # tolerancia (norma m��nima de delta)
iterm  <- 1000 # n�umero m�aximo de iteraciones
tolera <- 1 # inicializar tolera
itera  <- 0 # inicializar itera
histo  <- b # inicializar historial de iteraciones

while((tolera>tolm) && (itera < iterm)){
  eta   <- as.vector(X%*%b)

  # liga can�onica log(lamb) = eta
  lamb  <- exp(eta)
  z     <- eta + (y-lamb)/lamb

  aa    <- as.vector(solve(t(X*lamb)%*%X, t(X*lamb)%*%z))
  a2    <- solve(t(X*lamb)%*%X) %*% t(X*lamb) %*% z
  a3    <- b + solve(t(X*lamb)%*%X) %*% t(X) %*% matrix(y-lamb,n,1)


  delta <- aa-b
  b     <- aa
  tolera <- sqrt( sum(delta*delta) )
  histo  <- rbind(histo,b)
  itera  <- itera + 1 
}

lsat   <- sum( y*log(y) - y - lfactorial(y) ) # logv(saturado)
eta    <- as.vector(X%*%b)
lamb   <- exp(eta)
errstd <- sqrt(diag(solve(t(X*lamb)%*%X))) # 0.186877 0.045780 0.002659
lmax   <- sum( y*log(lamb) - y - lfactorial(y) ) # logv(mod inter�es)
lmax   <- sum( y*log(lamb) - lamb - lfactorial(y) ) # logv(mod inter�es)
lambN   <- mean(y)
lnull   <- sum( y*log(lambN) - y - lfactorial(y) ) # logv(nulo)
NullD   <- -2*( lnull - lsat ) # 872.2058 con 13-1=12 gl
ResD    <- -2*( lmax - lsat ) # 9.240248 con 13-3=10 gl
AIC     <- -2*lmax + 2*3 # 96.92358

##
# Usando glm
##

m1 <- glm(y~x+I(x^2),family="poisson")
summary(m1)

# Comparar cuadr�atico vs lineal
m0 <- summary( glm(y~x,poisson) )
m1 <- summary( glm(y~x+I(x^2),poisson) )
# Estad��stico de prueba
ji <- m0$deviance - m1$deviance
gl <- m0$df.residual - m1$df.residual
1-pchisq(ji,gl) # = 0 .... se rechaza fuertemente el modelo lineal


