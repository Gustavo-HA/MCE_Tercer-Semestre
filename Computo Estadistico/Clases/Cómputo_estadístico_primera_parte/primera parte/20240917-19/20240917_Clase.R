##
# Gráfica de una mezcla de 3 gaussianas
##

####

m1 = -3; s1 = 1; pi1 = .2
m2 = -1; s2 = 1; pi2 = .2
m3 = 1;  s3 = 2; pi3 = .6

g1 = dnorm(x,m1,s1)
g2 = dnorm(x,m2,s2)
g3 = dnorm(x,m3,s3)
mz = pi1*g1+pi2*g2+pi3*g3

plot(x,mz,type="l",lwd=4,col="blue",ylab="densidad",
main="Mezcla de 3 gaussianas")
lines(x,pi1*g1,col="cyan",lwd=4)
lines(x,pi2*g2,col="cyan",lwd=4)
lines(x,pi3*g3,col="cyan",lwd=4)
lines(x,mz,col="blue",lwd=4)

legend(1.2,.18,legend=c("N(-3,1), pi1=.2","N(-1,1),pi2=.2","N( 1,2), pi3=.6"), bty="n")



x = seq(-6,4,length=501)

m1 = -3; s1 = 1; pi1 = .4
m2 = -1; s2 = 1; pi2 = .3
m3 = 1;  s3 = 2; pi3 = .3

g1 = dnorm(x,m1,s1)
g2 = dnorm(x,m2,s2)
g3 = dnorm(x,m3,s3)
mz = pi1*g1+pi2*g2+pi3*g3

plot(x,mz,type="l",lwd=4,col="blue",ylab="densidad",
main="Mezcla de 3 gaussianas")
lines(x,pi1*g1,col="cyan",lwd=4)
lines(x,pi2*g2,col="cyan",lwd=4)
lines(x,pi3*g3,col="cyan",lwd=4)
lines(x,mz,col="blue",lwd=4)

legend(1.2,.18,legend=c("N(-3,1), pi1=.4","N(-1,1),pi2=.3","N( 1,2), pi3=.3"), bty="n")



##
# Simulación de 3 gausianas bivariadas
##

library(mvtnorm)
pis = c(1,1,1)/3
N = 600
m1 = c(3,3); m2 = c(6,4.5); m3 = c(9,6)
S1 = matrix(c(1,.7,.7,1),ncol=2)
S2 = matrix(c(1,-.7,-.7,1),ncol=2)
S3 = matrix(c(1,.7,.7,1),ncol=2)
m = 200

set.seed(84848)
d1 = rmvnorm(m, mean = m1, sigma = S1)
d2 = rmvnorm(m, mean = m2, sigma = S2)
d3 = rmvnorm(m, mean = m3, sigma = S3)
dat = rbind(d1,d2,d3)

aa = cbind( dmvnorm(dat, mean = m1, sigma = S1),
dmvnorm(dat, mean = m2, sigma = S2),
dmvnorm(dat, mean = m3, sigma = S3))

bb = t(pis*t(aa))
cc = bb/rowSums(bb)

rowSums(cc)

dd = rgb(cc[,1],cc[,2],cc[,3])

plot(d1[,1],d1[,2],col="red",pch=16,xlim=c(0,13),ylim=c(-1,10),
xlab="",ylab="",xaxt="n",yaxt="n")
points(d2[,1],d2[,2],col="green",pch=16)
points(d3[,1],d3[,2],col="blue",pch=16)

par(mfrow=c(2,1))
plot(dat[,1],dat[,2],col="black",pch=16,xlim=c(0,13),ylim=c(-1,10),xlab="",ylab="",xaxt="n",yaxt="n")
plot(dat[,1],dat[,2],col=dd,pch=16,xlim=c(0,13),ylim=c(-1,10),xlab="",ylab="",xaxt="n",yaxt="n")

par(mfrow=c(1,1))
plot(dat[,1],dat[,2],col="black",pch=16,xlim=c(0,13),ylim=c(-1,10))
grid()


##
# Algoritmo EM para 3 gaussinas
##

mv = matrix( c(1,5,6,8,9,9,1,0,0,1,1,0,0,1,1,0,0,1),ncol=2,byrow=T )

par(mfrow=c(1,1))
plot(dat[,1],dat[,2],col="black",pch=16,xlim=c(0,13),ylim=c(-1,10))
points(mv[1:3,1],mv[1:3,2],pch=16,col=2:4,cex=2)
grid()


mv[1,] # Media Pob 1
mv[2,] # Media Pob 2
mv[3,] # Media Pob 3
mv[4:5,] # Mat Varianza y Cov de Pob1
mv[6:7,] # Mat Varianza y Cov de Pob2
mv[8:9,] # Mat Varianza y Cov de Pob3

pis = c(1,1,1)/3 # Equiprobable cada población

delta = .001 # tolerancia
iterM = 1000 # número máximo de iteraciones
tolera = 1 # inicializar tolera
itera = 1 # inicializar itera

histo = array(0,dim=c(9,2,iterM)) # inicializar historial de iteraciones
histo[,,1] = mv

tra = matrix(0,iterM+1,6) # Trayectoria de los centroides
tra[1,] = c(mv[1,],mv[2,],mv[3,])

tra[1:4,]

LV = rep(0,iterM)

while( (tolera>delta) & (itera<iterM) ){

itera = itera + 1
mvold = mv

aa = cbind( dmvnorm(dat, mean = mv[1,], sigma = mv[4:5,]),
dmvnorm(dat, mean = mv[2,], sigma = mv[6:7,]),
dmvnorm(dat, mean = mv[3,], sigma = mv[8:9,]))

bb = t(pis*t(aa))

cc = bb/rowSums(bb)
Nr = colSums(cc)

pis = colMeans(cc)/sum(colMeans(cc))

for(i in 1:3){
  mv[i,] = colSums(cc[,i]*dat)/Nr[i]
  X = t(t(dat) - mv[i,])
  mv[(2*(i+1)):(2*(i+1)+1),] = (t(cc[,i]*X)%*%X)/Nr[i]
}

# cbind(mvold,mv)
  
  LV[itera] = sum(log(rowSums(bb)))

  tra[itera,] = c(mv[1,],mv[2,],mv[3,])

  histo[,,itera] = mv
  tolera = sum((mv-mvold)^2)
} # converge en 40 iteraciones


plot(LV[3:itera],type="l",lwd=2,col="blue",
main="Valor de Logverosimilitud en cada iteración")

aa = cbind( dmvnorm(dat, mean = mv[1,], sigma = mv[4:5,]),
dmvnorm(dat, mean = mv[2,], sigma = mv[6:7,]),
dmvnorm(dat, mean = mv[3,], sigma = mv[8:9,]))
bb = t(pis*t(aa))
cc = bb/rowSums(bb)
dd = rgb(cc[,1],cc[,2],cc[,3])
par(mar=c(0,0,0,0))
plot(dat[,1],dat[,2],col=dd,pch=16,xlim=c(0,13),ylim=c(-1,10),
xlab="",ylab="",xaxt="n",yaxt="n")
xm = c(3,6,9)
ym = c(3,4.5,6)
points(xm,ym,pch=".",cex=15,col="yellow")
lines(tra[1:itera,1],tra[1:itera,2],lwd=2)
lines(tra[1:itera,3],tra[1:itera,4],lwd=2)
lines(tra[1:itera,5],tra[1:itera,6],lwd=2)
points(mv[1:3,1],mv[1:3,2],pch=16)

##
# Uso de mclust
##

library(mclust)
out = Mclust(dat)
summary(out)

tt = apply(out$z,1,which.max) # prob. de pertenencia a cada grupo
yy = c(rep(1,200),rep(2,200),rep(3,200))
table(yy,tt)

#----------------------------------------------------
#Gaussian finite mixture model fitted by EM algorithm
#----------------------------------------------------
#Mclust EEV (ellipsoidal, equal volume and shape) model with 3 components:
#log-likelihood n df BIC ICL
#-2090.746 600 13 -4264.651 -4315.36
#Clustering table:
#1 2 3
#203 198 199

es = estep(modelName = "EEV", data = dat, parameters = out$parameters)
tt = apply(es$z,1,which.max) # prob. de pertenencia a cada grupo
yy = c(rep(1,200),rep(2,200),rep(3,200))
table(tt,yy)

# 1 2 3
#1 193 10 0
#2 7 186 5
#3 0 4 195



