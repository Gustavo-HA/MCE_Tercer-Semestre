
# Tabla 3x3
nn <- 9
n3 <- 3
ii <- rep(1,nn)
x1 <- kronecker(diag(n3),rep(1,n3))
x2 <- kronecker(rep(1,n3),diag(n3))
X  <- cbind(ii,x1,x2)

# Tabla 2x2

nn <- 4
n3 <- 2
ii <- rep(1,nn)
x1 <- kronecker(diag(n3),rep(1,n3))
x2 <- kronecker(rep(1,n3),diag(n3))

X <- cbind(Int=ii,Placebo=x1[,-2],Infarto=x2[,-2])
Y <- c(189,10845,104,10933)
N <- sum(Y)
cbind(Y,X)

wP <- tapply(Y,X[,"Placebo"],sum)/N
wI <- tapply(Y,X[,"Infarto"],sum)/N

mP <- match(X[,"Placebo"],names(wP))
mI <- match(X[,"Infarto"],names(wI))

Muij <- wP[mP]*wI[mI]*N

lMuij <- log(Muij)

RR <- lm(lMuij~-1+X)
summary(RR)

########################################
########################################

II  <- JJ <- KK <- 2
obs <- esp <- array(dim=c(II,JJ,KK))
obs[,,1] <- matrix( c(911,3,44,2), ncol=2 )
obs[,,2] <- matrix( c(538,43,456,279), ncol=2 )

for(i in 1:II){
for(j in 1:JJ){
for(k in 1:KK){
esp[i,j,k] <- sum( obs[i,,k] )*sum( obs[,j,k] )/sum( obs[,,k] ) }}}

G2 <- 2*sum(obs*log(obs/esp)) # 187.7543
X2 <- sum( ((obs-esp)^2)/esp ) # 177.6149

pG2 <- 1-pchisq(G2,2)
pX2 <- 1-pchisq(X2,2)

# Alternativamente, usando library(MASS)
dat <- data.frame(expand.grid(
marihuana = factor( c("Si","No"),levels=c("No","Si") ),
tabaco = factor( c("Si","No"),levels=c("No","Si") ),
alcohol = factor( c("Si","No"),levels=c("No","Si") )),
frec = c(911,538,44,456,3,43,2,279))

library(MASS)
outXZ.YZ <- loglm(frec ~ alcohol + tabaco + marihuana + alcohol*marihuana + tabaco*marihuana,
data=dat,param=T,fit=T)

outXZ.YZ; 
fitted(outXZ.YZ)

model.matrix(outXZ.YZ,data=dat)

## ajustando todos los modelos de tabla I x J x K
ff <- c("alcohol + tabaco + marihuana")
gg <- c(
"alcohol*tabaco",
"alcohol*marihuana",
"tabaco*marihuana",
"alcohol*tabaco + alcohol*marihuana",
"alcohol*tabaco + tabaco*marihuana",
"alcohol*marihuana + tabaco*marihuana",
"alcohol*tabaco + alcohol*marihuana + tabaco*marihuana",
"alcohol*tabaco+alcohol*marihuana+tabaco*marihuana+alcohol*tabaco*marihuana")

tt <- matrix(0,9,4)
colnames(tt) <- c("G2","X2","gl","pvalor")
out <- loglm(frec ~ alcohol+tabaco+marihuana,data=dat,param=T,fit=T)
tt[1,] <- c(out$lrt,out$pearson,out$df,1-pchisq(out$lrt,out$df))

for(j in 1:8){# j <- 8
fmla <- as.formula(paste("frec ~",ff,"+",gg[j]))
out <- loglm(fmla,data=dat,param=T,fit=T)
tt[j+1,] <- c(out$lrt,out$pearson,out$df,1-pchisq(out$lrt,out$df)) }

modelo <- c("X Y Z", "XY", "XZ", "YZ", "XY XZ", "XY YZ", "XZ YZ",
"XY XZ YZ", "XY XZ YZ XYZ")
tt <- data.frame(modelo,round(tt,2))
tt


################################################################################

dat <- read.csv("Cancer_China.csv")
dSi <- cbind(dat[,1:2],cancer="Si",frec=dat[,3])
dNo <- cbind(dat[,1:2],cancer="No",frec=dat[,4])
dat <- rbind(dSi,dNo)

loglm(formula=frec ~ fuma + cancer + ciudad + fuma*cancer +
fuma*ciudad + cancer*ciudad, data=dat, param = T, fit = T)


