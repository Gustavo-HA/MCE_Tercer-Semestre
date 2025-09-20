##
# ST1 
##

ST <- scan("Datos_ST1.txt") # ARMA(3,1)

par(mfrow=c(3,1))
plot.ts(ST)
acf(ST)
pacf(ST)

ari <- arima(ST,order=c(4,0,0),include.mean=T)
tsdiag(ari)

res <- ari$res
par(mfrow=c(2,1))
acf(res)
pacf(res)

ari <- arima(ST,order=c(3,0,1),include.mean=T)
tsdiag(ari)

res <- ari$res
par(mfrow=c(2,1))
acf(res)
pacf(res)


AR <- 0:5
MA <- 0:5
nAR <- length(AR)
nMA <- length(MA)

MAIC <- matrix(0,nAR,nMA)
for(iar in 1:nAR){
  for(ima in 1:nMA){# iar <- 1; ima <- 2
    if( (iar == 1) & (ima == 1)){ next }
    
    print(c(iar,ima))

    ari <- arima(ST,order=c(AR[iar],0,MA[ima]),include.mean=T)
    MAIC[iar,ima] <- ari$aic
    
  }
}

dimnames(MAIC) <- list(paste("AR_",AR,sep=""),paste("MA_",MA,sep=""))

MAIC[1,1] <- Inf

CC <- which(MAIC == min(MAIC),T)
AR[CC[1]]
MA[CC[2]]











ari <- arima(ST,order=c(2,0,1),include.mean=T)
ari
tsdiag(ari)


ari <- arima(ST,order=c(3,0,1),include.mean=T)
ari
tsdiag(ari)


##
# ST2
##

ST <- scan("Datos_ST2.txt") # ARMA(0,2)

par(mfrow=c(3,1))
plot.ts(ST)
acf(ST)
pacf(ST)

ari <- arima(ST,order=c(0,0,2),include.mean=T)
ari
tsdiag(ari)

ari <- arima(ST,order=c(0,0,3),include.mean=T)
ari
tsdiag(ari)


AR <- 0:5
MA <- 0:5
nAR <- length(AR)
nMA <- length(MA)

MAIC <- matrix(0,nAR,nMA)
for(iar in 1:nAR){
  for(ima in 1:nMA){# iar <- 1; ima <- 2
    if( (iar == 1) & (ima == 1)){ next }
    
    print(c(iar,ima))

    ari <- arima(ST,order=c(AR[iar],0,MA[ima]),include.mean=T)
    MAIC[iar,ima] <- ari$aic
    
  }
}

dimnames(MAIC) <- list(paste("AR_",AR,sep=""),paste("MA_",MA,sep=""))

MAIC[1,1] <- Inf

CC <- which(MAIC == min(MAIC),T)
AR[CC[1]]
MA[CC[2]]


##
# ST3
##

ST <- scan("Datos_ST3.txt") # ARMA(0,3)

par(mfrow=c(3,1))
plot.ts(ST)
acf(ST)
pacf(ST)

ari <- arima(ST,order=c(0,0,3),include.mean=T)
ari
tsdiag(ari)

ari <- arima(ST,order=c(1,0,3),include.mean=T)
ari
tsdiag(ari)



AR <- 0:5
MA <- 0:5
nAR <- length(AR)
nMA <- length(MA)

MAIC <- matrix(0,nAR,nMA)
for(iar in 1:nAR){
  for(ima in 1:nMA){# iar <- 1; ima <- 2
    if( (iar == 1) & (ima == 1)){ next }
    
    print(c(iar,ima))

    ari <- arima(ST,order=c(AR[iar],0,MA[ima]),include.mean=T)
    MAIC[iar,ima] <- ari$aic
    
  }
}

dimnames(MAIC) <- list(paste("AR_",AR,sep=""),paste("MA_",MA,sep=""))

MAIC[1,1] <- Inf

CC <- which(MAIC == min(MAIC),T)
AR[CC[1]]
MA[CC[2]]


##
# ST4
##

ST <- scan("Datos_ST4.txt") # ARMA(2,0)

par(mfrow=c(3,1))
plot.ts(ST)
acf(ST)
pacf(ST)

ari <- arima(ST,order=c(2,0,0),include.mean=T)
ari
tsdiag(ari)

ari <- arima(ST,order=c(2,0,1),include.mean=T)
ari
tsdiag(ari)


AR <- 0:5
MA <- 0:5
nAR <- length(AR)
nMA <- length(MA)

MAIC <- matrix(0,nAR,nMA)
for(iar in 1:nAR){
  for(ima in 1:nMA){# iar <- 1; ima <- 2
    if( (iar == 1) & (ima == 1)){ next }
    
    print(c(iar,ima))

    ari <- arima(ST,order=c(AR[iar],0,MA[ima]),include.mean=T)
    MAIC[iar,ima] <- ari$aic
    
  }
}

dimnames(MAIC) <- list(paste("AR_",AR,sep=""),paste("MA_",MA,sep=""))

MAIC[1,1] <- Inf

CC <- which(MAIC == min(MAIC),T)
AR[CC[1]]
MA[CC[2]]

