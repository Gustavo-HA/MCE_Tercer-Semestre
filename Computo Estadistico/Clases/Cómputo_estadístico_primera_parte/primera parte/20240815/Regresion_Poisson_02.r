install.packages("maps")

library(maps)

lung  <- read.table("MNlung.txt", header=TRUE, sep="\t")
radon <- read.table("MNradon.txt", header=TRUE)

MNmap <- function(data,ncol=5,figmain="",digits=5,type="e",lower=NULL, upper=NULL,udig=2){
  
  if (is.null(lower)) lower <- min(data)
  
  if (is.null(upper)) upper <- max(data)
  
  if (type=="q"){
    p <- seq(0,1,length=ncol+1)
    br <- round(quantile(data,probs=p),2)
  }

  if (type=="e"){
    br <- round(seq(lower,upper,length=ncol+1),udig)
  }

  shading <- gray((ncol-1):0/(ncol-1))

  data.grp <- findInterval(data,vec=br,rightmost.closed=T,all.inside=T)

  data.shad <- shading[data.grp]

  map("county", "minnesota", fill=TRUE, col=data.shad)

  leg.txt<-paste("[",br[ncol],",",br[ncol+1],"]",sep="")

  for(i in (ncol-1):1){
    leg.txt<-append(leg.txt,paste("[",format(br[i],digits=2), ",", format(br[i+1],digits=2),")",sep=""),)
  }

  leg.txt<-rev(leg.txt)

  legend(-91.9,46.7, legend=leg.txt,fill=shading,bty="n",ncol=1,text.width=1)
  
  title(main=figmain,cex=1.5); invisible() 
}

Obs <- lung[,"obs.M"] + lung[,"obs.F"]
Exp <- lung[,"exp.M"] + lung[,"exp.F"]
SMR <- Obs/Exp

.savedPlots <- NULL
windows(800,600,,T)

par(mfrow=c(1,1),mar=c(1,1,1,1)+.1) # bottom/left/top/right
MNmap(SMR, ncol=8, type="e", figmain="SMR", lower=min(SMR), upper=max(SMR))

rad.avg <- rep(0, length(Obs))
for(i in 1:length(Obs)) { 
  rad.avg[i] <- mean(radon[radon$county==i,2],na.rm=T)
}
rad.avg[is.na(rad.avg)] <- 0

MNmap(rad.avg,ncol=8,type="e",figmain="Rad�n",lower=min(rad.avg),upper=max(rad.avg),udig=1)

plot(Obs/Exp~rad.avg,xlab="Average radon (pCi/liter)",ylab="SMR", pch=16)
lines(lowess(Obs/Exp~rad.avg), lwd=3, col="red")

poismod <- glm(Obs~offset(log(Exp))+rad.avg,family="poisson")
summary(poismod)



