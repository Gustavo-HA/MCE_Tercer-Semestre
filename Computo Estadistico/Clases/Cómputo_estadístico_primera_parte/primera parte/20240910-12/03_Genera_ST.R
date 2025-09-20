
set.seed(2021)

# ARMA(2,1)
ST <- arima.sim(n = 1000, list(ar = c(0.70, -0.5), ma = c(0.25)), sd = 1)
write.table(ST,"Datos_ST1.txt",row.names=F,col.names=F)

# ARMA(0,2)
ST <- arima.sim(n = 1000, list(ma = c(0.50,0.25)), sd = 1)
write.table(ST,"Datos_ST2.txt",row.names=F,col.names=F)

# ARMA(1,3)
ST <- arima.sim(n = 1000, list(ar = c(0.50), ma = c(0.50,-0.25,0.125)), sd = 1)
write.table(ST,"Datos_ST3.txt",row.names=F,col.names=F)

# ARMA(2,0)
ST <- arima.sim(n = 1000, list(ar = c(0.50,0.25)), sd = 1)
write.table(ST,"Datos_ST4.txt",row.names=F,col.names=F)


          