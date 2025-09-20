##
# BootStrap no parámetrico básico
##

set.seed(4347)
n = 40
med = 10
des = 1
data = rnorm(n, med, des)

xb = mean(data)       # 9.770336
err = sd(data)/sqrt(n)# 0.1405979

B   = 5000
xbs = rep(0,B)
for(i in 1:B){
  sel    = sample(1:n, size=n, replace=TRUE)
  xx     = data[sel]
  xbs[i] = mean(xx)
}

sd(xbs) # 0.1386355 = error estándar bootstrap
err     # 0.1405979 = error estándar

sd(xbs) - err


##
# BootStrap: Pearson coeficiente de correlación
##

set.seed(9320)
lsat = c(576,635,558,578,666,580,555,661,651,605,653,575,545,572,594)
gpa  = c(339,330,281,303,344,307,300,343,336,313,312,274,276,288,296)
n    = length(gpa)

plot(lsat,gpa,pch=16,col=2)

robs = cor(lsat,gpa) # correlación observada = 0.7763745

# Cómo estimar su error estándar?
B = 5000
b = rep(0,B)
for(i in 1:B){
  sel = sample(1:n,size=n,replace=TRUE)
  b[i] = cor(lsat[sel],gpa[sel])
}

s = sd(b) # 0.1354122


##
# Bootstrap Paramétrico
##

# Bootstrap Paramétrico.
set.seed(9320)
lsat = c(576,635,558,578,666,580,555,661,651,605,653,575,545,572,594)
gpa  = c(339,330,281,303,344,307,300,343,336,313,312,274,276,288,296)
robs = cor(lsat,gpa) # correlaci´on observada = 0.7763745
plot(lsat,gpa,xlim=c(530,680),ylim=c(260,360),pch=16)

# (razonablemente como una normal bivariada)

X  = cbind(lsat,gpa)
mu = colMeans(X)
Sig = var(X)

# Supongamos que la población es: N(mu,Sig), entonces, una forma
# de estimar el comportamiento distribucional de Ro, es muestrear
# de esta población.

library(mvtnorm)
n = length(gpa)
B = 5000
ros = rep(0,B)
for(i in 1:B){
  mues = rmvnorm(n=n,mean=mu,sigma=Sig)
  ros[i] = cor(mues[,1],mues[,2])
}

s = sd(ros)
# Estimación bootstrap del error estándar (bootstrap paramétrico)
# s = 0.119026

##
# Distribución del coeficiente de correlación de Pearson
##

# install.packages("hypergeo")

library(hypergeo)
M = 201
rr = seq(0,1,length=M)
ff = rep(0,M)
for(i in 1:M){
aa = ((n-2)*gamma(n-1))/(sqrt(2*pi)*gamma(n-.5))
bb = ((1-robs^2)^((n-1)/2)) / ((1-robs*rr[i])^(n-1.5))
cc = (1-rr[i]^2)^((n-4)/2)
dd = hypergeo(.5,.5,(2*n-1)/2,(robs*rr[i]+1)/2)
ff[i] = aa*bb*cc*dd }

par(mfrow=c(1,2),mar=c(4,4,1,1))
hist(ros,main="Réplicas (Paramétricas) del Coef. de Corr.",
cex.main=.9,xlab="r(*)",ylab="",cex.axis=.8,mgp=c(1.5,.5,0),
col="cyan",nclass=20,xlim=c(0,1),probability=TRUE,ylim=c(0,4.5))
points(robs,.1,pch=16,col="red")
lines(rr,ff,lwd=2)

hist(b,main="Réplicas (No Paramétricas) del Coef. de Corr.",
cex.main=.9,xlab="r(*)",ylab="",cex.axis=.8,mgp=c(1.5,.5,0),
col="cyan",nclass=20,xlim=c(0,1),probability=TRUE,ylim=c(0,4.5))

points(robs,.1,pch=16,col="red")
lines(rr,ff,lwd=2)


##
# Estmación Plug-in
##

xx = seq(-2,2,len=51)
yy = pnorm(xx)

plot(xx,yy,lwd=2,col="blue",type="l")

segments(-2,0,-sqrt(2),0,lwd=2,col="red")
segments(-sqrt(2),0,-sqrt(2),1/4,lwd=2,col="red",lty=2)
segments(-sqrt(2),1/4,0,1/4,lwd=2,col="red")
segments(0,1/4,0,3/4,lwd=2,col="red",lty=2)
segments(0,3/4,sqrt(2),3/4,lwd=2,col="red")
segments(sqrt(2),3/4,sqrt(2),1,lwd=2,col="red",lty=2)
segments(sqrt(2),4/4,2,4/4,lwd=2,col="red")

set.seed(83838)
dat = c(3,5)
B  = 1000
xb = rep(0,B)
for(i in 1:B){
  sel = sample(1:2,size=2,replace=TRUE)
  dd = dat[sel]
  xb[i] = mean(dd)
}

zz = sqrt(2)*(xb-mean(dat))
  
table(zz)/length(zz)



##
# Intervalos de confianza bootstrap
##

set.seed(9320)
lsat = c(576,635,558,578,666,580,555,661,651,605,653,575,545,572,594)
gpa = c(339,330,281,303,344,307,300,343,336,313,312,274,276,288,296)
robs = cor(lsat,gpa) # correlaci´on observada = 0.7763745
B = 5000
b = rep(0,B)
for(i in 1:B){
sel = sample(1:n,size=n,replace=TRUE)
b[i] = cor(lsat[sel],gpa[sel]) }
s = sd(b)

# s = 0.1333642
# Intervalos de Confianza, 90%
# M´etodo de percentiles:

icn = quantile(b,p=c(.05,.95))
#5% 95%
# 0.5200723 0.9456298

# Método Básico:
lininf = 2*robs-icn[2] # 0.607
limsup = 2*robs-icn[1] # 1.033

##
# Intervalos de confianza BCa
##

set.seed(9320)
lsat = c(576,635,558,578,666,580,555,661,651,605,653,575,545,572,594)
gpa = c(339,330,281,303,344,307,300,343,336,313,312,274,276,288,296)
n = length(gpa)
robs = cor(lsat,gpa) # correlaci´on observada = 0.7763745
B = 5000; b = rep(0,B)
for(i in 1:B){
sel = sample(1:n,size=n,replace=TRUE)
b[i] = cor(lsat[sel],gpa[sel]) }
z0 = qnorm(mean(b < robs))
rjack = rep(0,n)
for(i in 1:n){ rjack[i] = cor(lsat[-i],gpa[-i]) }
rm = mean(rjack)
a = sum( (rm-rjack)^3 ) / (6*( sum( (rm-rjack)^2 ) )^(1.5))
aen2 = .05
zal = qnorm(aen2); zaln = qnorm(1-aen2)
alf1 = pnorm(z0 + (z0+zal)/(1-a*(z0+zal)))
alf2 = pnorm(z0 + (z0+zaln)/(1-a*(z0+zaln)))
inter = quantile(b,probs=c(alf1,alf2))

# 1.78833% 90.17267% 
# 0.4447953 0.9315602

