#setwd("~/groups/aggregator/trunk/test")

# Generate sliding normal
sliding.normal <- function(xx, yy) {
  table <- matrix(NA, nx+1, ny+1)
  table[1,1] <- 'ddp1'
  table[1,-1] <- yy
  table[-1,1] <- xx
  for (ii in 1:length(xx)) {
    cdf <- c(pnorm(yy, xx[ii] / 8), 1)
    table[ii+1,-1] <- diff(cdf)
  }

  return(table)
}

as.logs <- function(table) {
  table[1,1] <- 'ddp2'
  table[-1,-1] <- log(as.numeric(table[-1,-1]))

  return(table)
}

# Generate constant uniform
constant.uniform <- function(xx, yy, min, max) {
  table <- matrix(NA, nx+1, ny+1)
  table[1,1] <- 'ddp1'
  table[1,-1] <- yy
  table[-1,1] <- xx

  nn = sum(yy >= min & yy < max)
  for (ii in 1:length(xx))
    table[ii+1,-1] <- (yy >= min & yy < max) / nn

  return(table)
}

nx <- 10
ny <- 10
xx <- seq(0, 40, length.out=nx)
yy <- seq(-10, 10, length.out=ny)

table <- sliding.normal(xx, yy)
write.table(table, "ddp1.csv", row.names=F, col.names=F, quote=F, sep=",")
write.table(as.logs(table), "ddp2.csv", row.names=F, col.names=F, quote=F, sep=",")
write.table(constant.uniform(xx, yy, 0, 5), "ddp3.csv", row.names=F, col.names=F, quote=F, sep=",")
