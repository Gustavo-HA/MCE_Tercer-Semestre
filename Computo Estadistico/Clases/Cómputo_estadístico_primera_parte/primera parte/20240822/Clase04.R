n  <- 20
ld <- 0:5
X <- rbind(cbind(1,0,ld,0), cbind(1,1,ld,ld))

y <- c(1,4,9,13,18,20,0,2,6,10,12,16)

z <- y/n # datos son proporciones
b <- c(-1,0,0,0) # valores iniciales
tolm <- 1e-4 # tolerancia (norma minima de delta)
iterm <- 100 # numero maximo de iteraciones
tolera <- 1 # inicializar tolera
itera <- 0 # inicializar itera
histo <- b # inicializar historial de iteraciones
# (No es coincidencia que es el mismo c´ogigo que antes)
while( (tolera>tolm)&(itera<iterm) ){
eta <- as.vector(X%*%b)
pi <- 1/(1+exp(-eta))

U <- n*pi*(1-pi)

wy <- eta + (z-pi)/(pi*(1-pi))
aa <- (lm( wy ~ -1+X, weights=U ))$coeff # min cuad ponderados
delta <- aa-b
b <- aa
tolera <- sqrt( sum(delta*delta) )
histo <- rbind(histo,b)
itera <- itera + 1 }
# aa <- solve(t(X)%*%(U*X), t(X)%*%(U*wy)) # min cuad ponderados

## usando glm de R
yy  <- cbind(n*z,n-n*z)
out <- glm( yy ~ X[,-1], family=binomial(link="logit"))
summary(out)

# Replicamos los calculos anteriores de glm(), expl´ýcitamente:
pi  <- z
sel <- (pi != 0 & pi != 1)
aa  <- z[sel]*log(pi[sel]/(1-pi[sel]))+log(1-pi[sel])
lsat <- n*sum( aa ) + sum( lchoose(n,n*z) )
etai <- as.vector(X%*%b)
pf <- 1/(1+exp(-eta))
Uf <- n*pf*(1-pf)
errstd <- sqrt(diag(solve(t(X)%*%(U*X)))) # 0.548 0.778 0.212 0.270

pbar <- mean(z)
lmax <- n*sum( z*etai-log(1+exp(etai)) ) + sum( lchoose(n,n*z) )
lnull <- n*sum(z*log(pbar/(1-pbar))+log(1-pbar)) + sum(lchoose(n,n*z))
NullD <- -2*( lnull - lsat ) # 124.8756 con 13-1=12 gl
ResD <- -2*( lmax - lsat ) # 4.993727 con 12-4= 8 gl
AIC <- -2*lmax + 2*4 # 43.10413

# Grafica mortalidad orugas machos vs hembras
xx <- c(ld,ld)
plot(xx,z, xlab="Log2(dosis)",type="n",mgp=c(1.5,.5,0),cex.axis=.8)
points(ld,z[1:6], pch="M", cex=.7)
points(ld,z[7:12], pch="H", cex=.7)
dd <- seq(0,5,length=100)
lines(dd,1/(1+exp(-b[1]-b[3]*dd)),col="blue",lwd=2,ylab="Prop. de M.")
lines(dd,1/(1+exp(-b[1]-b[2]-(b[3]+b[4])*dd)), col="red", lwd=2)

# Ajuste del modelo sin interacci´on
n <- 20; ld <- 0:5
X <- rbind(cbind(1,0,ld), cbind(1,1,ld))
y <- c(1,4,9,13,18,20,0,2,6,10,12,16)
yy <- cbind(y,n-y)
out0 <- glm( yy ~ X[,-1], family=binomial(link="logit"))
summary(out0)

#Call:
#glm(formula = yy ~ X[, -1], family = binomial(link = "logit"))
#
#Deviance Residuals: 
#     Min        1Q    Median        3Q       Max  
#-1.10540  -0.65343  -0.02225   0.48471   1.42944  
#
#Coefficients:
#            Estimate Std. Error z value Pr(>|z|)    
#(Intercept)  -2.3724     0.3855  -6.154 7.56e-10 ***
#X[, -1]      -1.1007     0.3558  -3.093  0.00198 ** 
#X[, -1]ld     1.0642     0.1311   8.119 4.70e-16 ***
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#(Dispersion parameter for binomial family taken to be 1)
#
#    Null deviance: 124.8756  on 11  degrees of freedom
#Residual deviance:   6.7571  on  9  degrees of freedom
#AIC: 42.867
#
#Number of Fisher Scoring iterations: 4
#

b  <- summary(out0)$coeff[,1]
pi <- z
sel <- (pi != 0 & pi != 1)
aa <- z[sel]*log(pi[sel]/(1-pi[sel]))+log(1-pi[sel])
lsat <- n*sum( aa ) + sum( lchoose(n,n*z) )
etai <- as.vector(X%*%b)
pf <- 1/(1+exp(-eta))
Uf <- n*pf*(1-pf)
errstd <- sqrt(diag(solve(t(X)%*%(U*X)))) # 0.38551 0.35583 0.13108
pbar <- mean(z)

lmax <- n*sum( z*etai-log(1+exp(etai)) ) + sum( lchoose(n,n*z) )
lnull <- n*sum(z*log(pbar/(1-pbar))+log(1-pbar)) + sum(lchoose(n,n*z))
NullD <- -2*( lnull - lsat ) # 124.8756 con 13-1=12 gl
ResD0 <- -2*( lmax - lsat ) # 6.757064 con 12-3= 9 gl
AIC0 <- -2*lmax + 2*3 # 42.86747
# Comparaci´on del modelo con interacci´on (mod completo)
# contra un modelo sin interacci´on (mod reducido)
Delta <- ResD0 - ResD
pval <- 1-pchisq(Delta,4-3) # 0.1842 no hay suf. evidencia para rech
# el modelo sin interacci´on.
# Gr´afica mortalidad orugas machos vs hembras

xx <- c(ld,ld)
plot(xx,z, xlab="Log2(dosis)", type="n", mgp=c(1.5,.5,0),
cex.lab=.8, cex.axis=.8, main="Modelo sin interacción")
points(ld,z[1:6], pch="M", cex=.7)
points(ld,z[7:12], pch="H", cex=.7)
dd <- seq(0,5,length=100)
lines(dd,1/(1+exp(-b[1]-b[3]*dd)),col="blue",lwd=2, ylab="Prop. de M.")
lines(dd,1/(1+exp(-b[1]-b[2]-b[3]*dd)), col="red", lwd=2)
