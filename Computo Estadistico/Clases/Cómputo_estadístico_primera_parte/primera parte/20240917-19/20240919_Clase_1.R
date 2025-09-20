####
##  Caso 1
####

h <- function(x){
  (x-1)^2 # funcion objetivo
}

x0 <- 0.5
h0 <- h(x0)

ua <- 0.1
n  <- 20000
r <- 1
z <- matrix(0,n,3)
z[1,] <- c(x0,h0,r)

set.seed(2021)
for( i in 2:n ){# i <- 2
  ti <- .1/(log(1+i)) # parametro clave: "temperatura"
  
  xt <- runif(1,x0-ua,x0+ua)  
  ht <- h(xt)
  dh <- h0 - ht
  
  r <- min(exp(dh/ti),1)
  if(runif(1) < r){
    x0 <- xt
    h0 <- ht
    z[i,] <- c(xt,ht,r)
  } else {
    z[i,] <- c(x0,h0,r)
  }
  
}

par(mfrow=c(2,1),mar=c(3.1,3.1,2,1))
xs <- seq(.4,1.75,length=200)
plot(xs,h(xs),type="l",ylab="h(x)",xlab="",mgp=c(2,.5,0),xlim=c(.4,1.75))
rug(z[,1])

hist(z[,1],main="",col="cyan",xlab="x",mgp=c(2,.5,0),xlim=c(.4,1.75))

cmin <- which.min(z[,2])
z[cmin,]





####
##  Caso 2
####

h <- function(x,y){
(x*sin(20*y) + y*sin(20*x))^2 * cosh(sin(10*x)*x) +
(x*cos(10*y) - y*sin(10*x))^2 * cosh(cos(20*y)*y) }

x <- y <- seq(-1,1,length=100)
zi <- outer(x,y,"h")

persp(x,y,zi,theta=60,phi=30)
contour(x,y,zi)

h00 <- h(0,0)

set.seed(2021)

x0 <- 0.999
y0 <- 0.999

h0 <- h(x0,y0)
ua <- 0.1
n <- 5000
z <- matrix(0,n,4) ; z[1,] <- c(x0,y0,h0,1)
rem <- c()
for( i in 2:n ){
  ti <- 1/(log(1+i))
  xt <- runif(1,x0-ua,x0+ua)
  yt <- runif(1,y0-ua,y0+ua)
  if( abs(xt) > 1 | abs(yt) > 1 ){ 
    rem <- c(rem,i)
    next
  }
  ht <- h(xt,yt)
  dh <- h0 - ht
  r <- min(exp(dh/ti),1)
  if(runif(1) < r){
    x0 <- xt
    y0 <- yt
    h0 <- ht
    z[i,] <- c(xt,yt,ht,r)
  }else{
    z[i,] <- c(x0,y0,h0,r)
  }
}

z <- z[-rem,]
m <- nrow(z)

x <- y <- seq(-1,1,length=100)
zi <- outer(x,y,"h") ; par(mar=c(2,2,2,2))
image(x,y,zi,xaxt="n",yaxt="n",ylab="",xlab="",cex.main=1,
main="Trayectoria de Algoritmo Recocido Simulado")
for(i in 1:(m-1)){ segments(z[i,1],z[i,2],z[i+1,1],z[i+1,2]) }



cmin <- which.min(z[,3])
round(z[cmin,],6)


