
##
# MA(1)
##

set.seed(2021)
N  <- 10000
e0 <- rnorm(1,0,1)
et <- rnorm(N,0,1)

te1 <- 0.40
te2 <- 0.95

y1 <- y2 <- rep(0,N)

y1[1] <- et[1] + te1*e0
y2[1] <- et[1] + te2*e0

for(tt in 2:N){
  y1[tt] <- et[tt] + te1*et[tt-1]
  y2[tt] <- et[tt] + te2*et[tt-1]
}

par(mfrow=c(2,1))
plot.ts(y1,main="Teta = 0.40",col="steelblue",lwd=2)
plot.ts(y2,main="Teta = 0.95",col="orange",lwd=2)


par(mfrow=c(2,1))
acf1 <- acf(y1)
acf2 <- acf(y2)

par(mfrow=c(2,1))
pacf1 <- pacf(y1)
pacf2 <- pacf(y2)

par(mfrow=c(2,1))
acf1 <- acf(y1)
pacf1 <- pacf(y1)


yma1 <- y1
yma2 <- y2

yy <- y2

yr1 <- yy[3:N]
yr2 <- yy[2:(N-1)]
yr3 <- yy[1:(N-2)]
#cbind(3:N,2:(N-1),1:(N-2))

# autocorrelación
cor(yr1,yr2)
cor(yr1,yr3)

# autocorrelación parcial
lm(yr1~yr2)
lm(yr1~yr2+yr3)

ari <- arima(y1,order=c(1,0,0))
tsdiag(ari)
summary(ari)


##
# AR(1)
##

set.seed(2021)
N  <- 10000
et <- rnorm(N,0,1)

psi1 <- 0.40
psi2 <- 0.95

y1 <- y2 <- rep(0,N)
for(tt in 2:N){
  y1[tt] <- psi1 * y1[tt-1] + et[tt]
  y2[tt] <- psi2 * y2[tt-1] + et[tt]
}

par(mfrow=c(2,1))
plot.ts(y1,main="AR(1): Psi = 0.40",col="steelblue",lwd=2)
plot.ts(y2,main="AR(1): Psi = 0.95",col="orange",lwd=2)

par(mfrow=c(2,1))
acf1 <- acf(y1)
acf2 <- acf(y2)

par(mfrow=c(2,1))
pacf1 <- pacf(y1)
pacf2 <- pacf(y2)

par(mfrow=c(2,2))
acf1 <- acf(yma1,main="MA(1) 0.40")
acf2 <- acf(y1,main="AR(1) 0.40")
acf1 <- pacf(yma1,main="MA(1) 0.40")
acf2 <- pacf(y1,main="AR(1) 0.40")


par(mfrow=c(2,2))
acf1 <- acf(yma2,main="MA(1) 0.95")
acf2 <- acf(y2,main="AR(1) 0.95")
acf1 <- pacf(yma2,main="MA(1) 0.95")
acf2 <- pacf(y2,main="AR(1) 0.95")



yy <- y2

yr1 <- yy[3:N]
yr2 <- yy[2:(N-1)]
yr3 <- yy[1:(N-2)]
#cbind(3:N,2:(N-1),1:(N-2))

# autocorrelación
cor(yr1,yr2)
cor(yr1,yr3)

# autocorrelación parcial
lm(yr1~yr2)
lm(yr1~yr2+yr3)



##
# ARMA(3,2)
##

set.seed(2021)
N  <- 10000
et <- rnorm(N,0,1)

ar3 <- c(0.5,0.2,0.1)
ma2 <- c(0.5,0.25)

y1 <- rep(0,N)
for(tt in 4:N){
  y1[tt] <- ar3[1]*y1[tt-1] + ar3[2]*y1[tt-2] + ar3[3]*y1[tt-3] + ma2[1]*et[tt-1] + ma2[2]*et[tt-2] + et[tt]
}

plot.ts(y1)

par(mfrow=c(2,1))
acf(y1)
pacf(y1)


par(mfrow=c(2,1))
pacf1 <- pacf(y1)
pacf2 <- pacf(y2)

par(mfrow=c(2,2))
acf1 <- acf(yma1,main="MA(1) 0.40")
acf2 <- acf(y1,main="AR(1) 0.40")
acf1 <- pacf(yma1,main="MA(1) 0.40")
acf2 <- pacf(y1,main="AR(1) 0.40")


par(mfrow=c(2,2))
acf1 <- acf(yma2,main="MA(1) 0.95")
acf2 <- acf(y2,main="AR(1) 0.95")
acf1 <- pacf(yma2,main="MA(1) 0.95")
acf2 <- pacf(y2,main="AR(1) 0.95")




