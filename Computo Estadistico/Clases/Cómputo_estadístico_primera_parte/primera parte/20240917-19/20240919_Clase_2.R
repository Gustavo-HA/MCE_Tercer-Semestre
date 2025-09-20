# Metropolis-Hastings
# Tomado de Robert & Casella (2010)
# Introducing Monte Carlo Methods with R, p´ag 179

library(mvtnorm)

data(cars) # 50 x 2
str(cars) # cars$speed cars$dist

y  <- cars$dist
x1 <- cars$speed
x2 <- x1^2
n  <- length(y)
X  <- cbind(1,x1,x2)

summary(lm(y~-1+X))

tg <- solve( t(X)%*%X, t(X)%*%y )
s2 <- sum( (y-X%*%tg)^2 ) / (n-3)
VV <- s2 * solve( t(X)%*%X )

# núcleo de posterior
ff   <- function(teta){
coe  <- teta[1:3]
sig2 <- teta[4]
return(( 1/( (sig2)^(n/2) ) )*exp(-sum( (y-X%*%coe)^2 )/(2*sig2))) }

# núcleo de propuesta
gg   <- function(teta){
coe  <- teta[1:3]
sig2 <- teta[4]
aa   <- dmvnorm( coe, mean=tg, sigma=VV )
bb   <- dgamma( 1/sig2, shape=(n-3)/2, rate=(n-3)*s2/2 )
return(aa*bb) }

# simulación de propuesta
geneq <- function(){
aa <- rmvnorm( 1, mean=tg, sigma=VV )
bb <- rgamma( 1, shape=(n-3)/2, rate=(n-3)*s2/2 )
tetasim <- c(aa,1/bb)
return(tetasim) }

# La parte medular del Metropolis-Hastings es la siguiente
M    <- 10000
teta <- matrix(0,M+1,4)
teta[1,] <- c(tg,s2)

for(t in 1:M){# t <- 1
  tetap <- geneq()
  kk    <- ff(tetap)*gg(teta[t,])/(ff(teta[t,])*gg(tetap))
  if( runif(1) < kk ){
    teta[t+1,]=tetap
  }else{ 
    teta[t+1,]=teta[t,]
  }
}

teta2   <- teta[-(1:5000),]
medpost <- colMeans(teta2)
c(tg,s2)

plot(x1,y,xlab="Velocidad",ylab="Distancia",pch=20,type="n")
set.seed(8958)
M   <- 50
sel <- sample(1:5000,size=M,replace=FALSE)
for(i in 1:M){
  lines(x1,teta2[sel[i],1]+teta2[sel[i],2]*x1+teta2[sel[i],3]*x2,lwd=1,col="cyan")
}
points(x1,y,pch=20)
lines(x1,tg[1]+tg[2]*x1+tg[3]*x2,lwd=2,col="red")

legend("topleft",inset=0.03,col=c("cyan","red"),lwd=c(1,2),
legend=c("curvas simuladas de posterior","curva estimada"))

par(mfrow=c(2,2),mar=c(2,4,2,2))
plot(teta2[4000:5000,1],type="l",ylab="a"); abline(h=medpost[1],col="white",lwd=2)
plot(teta2[4000:5000,2],type="l",ylab="b"); abline(h=medpost[2],col="white",lwd=2)
plot(teta2[4000:5000,3],type="l",ylab="c"); abline(h=medpost[3],col="white",lwd=2)
plot(teta2[4000:5000,4],type="l",ylab="s2"); abline(h=medpost[4],col="white",lwd=2)

par(mfrow=c(2,2),mar=c(2,4,2,2))
plot(teta2[,1],type="l",ylab="a"); abline(h=medpost[1],col="white",lwd=2)
plot(teta2[,2],type="l",ylab="b"); abline(h=medpost[2],col="white",lwd=2)
plot(teta2[,3],type="l",ylab="c"); abline(h=medpost[3],col="white",lwd=2)
plot(teta2[,4],type="l",ylab="s2"); abline(h=medpost[4],col="white",lwd=2)

##
# Inferencia predictiva
##

# Supongamos x = 22
xn <- 22
M  <- 301
yn <- seq(0,142.1333,length.out = M)

med <- teta2[,1:3]%*%c(1,xn,xn^2)
des <- sqrt(teta2[,4])

pred <- rep(0,M)
for(i in 1:M){# i <- 1
  pred[i] <- mean( dnorm(yn[i],mean=med,sd=des) )
}

plot(yn,pred)

intpred <- function(y){
  return( mean( dnorm(y,mean=med,sd=des) ) )
}

integrate(Vectorize(intpred),lower=0,upper=142.1333)
# 0.9999743 with absolute error < 2.1e-11

ymax <- yn[which.max(pred)]

# a prueba y error un cuantil adecuado de la predictiva (90%)
integrate(Vectorize(intpred),lower=0,upper=96.64335)$value # 0.9500937

cuantil2 <- 96.64335
mp <- yn[which.max(pred)]
cuantil1 <- mp - (cuantil2 - mp)
y1 <- pred[ which.min(abs(cuantil1-yn)) ]
y2 <- pred[ which.min(abs(cuantil2-yn)) ]

plot(yn,pred,type="n",lwd=2,col="blue", ylab="Densidad predictiva",
xlab="Distancia de frenado, para una velocidad de x = 22")
grid()
segments(ymax,-0.1,ymax,max(pred),col="red",lwd=2)
segments(cuantil1,-0.1,cuantil1,(y1+y2)/2,col="red",lwd=2)
segments(cuantil2,-0.1,cuantil2,(y1+y2)/2,col="red",lwd=2)
lines(yn,pred,lwd=2,col="blue")

plot(x1,y,xlab="Velocidad",ylab="Distancia",pch=20,type="p",
main="Intervalos (90%) para Dist, cuando Vel = 22")
segments(xn,cuantil1,xn,cuantil2,lwd=4,col="red")
points(xn,ymax,pch=16,col="yellow")
# Un intervalo para la media (no de predicción)

med <- teta2[,1:3]%*%c(1,xn,xn^2)

cuantm <- quantile(med,probs = c(.05,.50,.95))

segments(xn-.4,cuantm[1],xn-.4,cuantm[3],lwd=4,col="blue")
points(xn-.4,cuantm[2],pch=16,col="yellow")
legend("topleft",inset=0.03,legend=c("para Media","para Predicci´on"),
lwd=4, col=c("blue","red"))


##
# BootStrap no parámetrico básico
##

set.seed(4347)
n = 40
med = 10
des = 1
data = rnorm(n, med, des)
xb = mean(data) # 9.770336
err = sd(data)/sqrt(n)
B = 5000
xbs = rep(0,B)
for(i in 1:B){
sel = sample(1:n, size=n, replace=TRUE)
xx = data[sel]
xbs[i] = mean(xx) }

sd(xbs) # 0.1377988 = error estándar bootstrap
err     # 0.1405979 = error estándar

##
# BootStrap: Pearson coeficiente de correlación
##

set.seed(9320)
lsat = c(576,635,558,578,666,580,555,661,651,605,653,575,545,572,594)
gpa = c(339,330,281,303,344,307,300,343,336,313,312,274,276,288,296)
n = length(gpa)
robs = cor(lsat,gpa) # correlaci´on observada = 0.7763745
# Cómo estimar su error estándar?
B = 5000
b = rep(0,B)
for(i in 1:B){
sel = sample(1:n,size=n,replace=TRUE)
b[i] = cor(lsat[sel],gpa[sel]) }

s = sd(b) # 0.1333642

