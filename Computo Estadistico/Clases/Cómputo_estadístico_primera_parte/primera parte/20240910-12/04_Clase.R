dat <- read.table("Ventas.txt",sep="|",header=T)
nd  <- nrow(dat)

plot.ts(dat[,2],ylab="Unidades",main="Venta Nacional: Semanal",xlab="",xaxt="n")
axis(1,1:nd,dat[,1],las=2,cex.axis=0.5)

agr <- aggregate(dat[,2],list(substring(dat[,1],1,6)),sum)
na  <- nrow(agr)
plot.ts(agr[,2],ylab="Unidades",main="Venta Nacional: Mensual",xlab="",xaxt="n")
axis(1,1:na,agr[,1],las=2,cex.axis=0.5)

Y  <- dat[,"Uni"]
nY <- length(Y)

## Inter 
X <- rep(1,nY)

RR  <- lm(Y~-1+X)
RES <- RR$res

plot.ts(RES,ylab="Residuales",main="Modelo: Inter",xlab="",xaxt="n")

## Inter + Tend
X <- cbind(Inter=rep(1,nY),Tend=1:nY)

RR  <- lm(Y~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend",xlab="",xaxt="n")

## Inter + Tend + Estac

Fec <- dat[,"semana"]
Mes <- as.numeric(substring(Fec,5,6))
XMes <- matrix(0,nY,12)
for(i in 1:nY){
  XMes[i,Mes[i]] <- 1
}
XMes <- XMes[,-1]

X <- cbind(Inter=rep(1,nY),Tend=1:nY,XMes)

RR  <- lm(Y~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)

## Inter + Tend + Estac + Trafico

plot.ts(dat[,"trafico"],ylab="Trafico",main="Trafico",xlab="",xaxt="n")

colnames(XMes) <- paste("Mes_",2:12,sep="")

X <- cbind(Inter=rep(1,nY),Tend=1:nY,XMes,Trafico=dat[,"trafico"])
X[1:4,]

RR  <- lm(Y~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend + Traf",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)


## Inter + Tend + Estac + Trafico, Estandarizar por mismas tiendas

plot.ts(dat[,"tiendas_maduras"],ylab="Tiendas",main="Tiendas",xlab="",xaxt="n")

YMT <- Y/dat[,"tiendas_maduras"]

X <- cbind(Inter=rep(1,nY),Tend=1:nY,XMes,Trafico=dat[,"trafico"])
X[1:4,]

RR  <- lm(YMT~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend + Traf",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)


## Inter + Tend + Estac + Trafico, Estandarizar por mismas tiendas
## Aislando efectos atipicos

SS <- scale(RES)

Fec[SS > 2]
OutPos <- ifelse(SS > 2,1,0)

Fec[SS < -2]
OutNeg <- ifelse(SS < -2,1,0)

X <- cbind(Inter=rep(1,nY),Tend=1:nY,XMes,Trafico=dat[,"trafico"],OutPos,OutNeg)
X[1:4,]

RR  <- lm(YMT~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend + Traf + Out",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)


## Inter + Tend + Estac + Trafico, Estandarizar por mismas tiendas
## Aislando efectos atipicos + Precio

Precio <- dat[,"Val"]/dat[,"Uni"]

X <- cbind(Inter=rep(1,nY),Precio,Tend=1:nY,XMes,Trafico=dat[,"trafico"],OutPos,OutNeg)
X[1:4,]

RR  <- lm(YMT~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Inter + Tend + Traf + Out + Precio",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)


## Inter + Tend + Estac + Trafico, Estandarizar por mismas tiendas
## Aislando efectos atipicos + Precio
## Transformar la demanda cn Logaritmo para homogenizar la varianza

LYMT <- log(YMT)

X <- cbind(Inter=rep(1,nY),Precio=log(Precio),Tend=log(1:nY),XMes,
Trafico=log(dat[,"trafico"]),OutPos,OutNeg)
X[1:4,]


RR  <- lm(LYMT~-1+X)
RES <- RR$res
summary(RR)

plot.ts(RES,ylab="Residuales",main="Modelo: Log (Inter + Tend + Traf + Out + Precio)",xlab="",xaxt="n")
axis(1,1:nd,Fec,las=2,cex.axis=0.5)

par(mfrow=c(2,1))
acf(RES)
pacf(RES)

ari <- arima(RES,order=c(1,0,0))
tsdiag(ari)


# Estimación conjunta de Regresión con errores arma

ari <- arima(LYMT,order=c(1,0,1),xreg=X,include.mean=FALSE)
tsdiag(ari)
ari

ari <- arima(LYMT,order=c(1,0,1),seasonal=list(order=c(1,0,1),period=12),xreg=X,include.mean=FALSE)
