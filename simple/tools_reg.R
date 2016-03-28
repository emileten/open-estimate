### Simple Robust Standard Errors Library for R
## Usage:
## source("tools_reg.R")
## mod ~ lm(satell ~ width + factor(color)) # assuming we have these data
## get.coef.clust(mod, color)

### Calculate heteroskedastically-robust standard errors
## mod: the result of a call to lm()
##      e.g., lm(satell ~ width + factor(color))
## var: a list of indexes to extract only certain coefficients (default: all)
##      e.g., 1:2 [to drop FE from above]
## estimator: an estimator type passed to vcovHC (default: White's estimator)
## Returns an object of type coeftest, with estimate, std. err, t and p values
get.coef.robust <- function(mod, var=c(), estimator="HC") {
  require(sandwich, quietly = TRUE)
  require(lmtest, quietly = TRUE)

  ## Construct the covariance matrix
  chc <- vcovHC(mod, type=estimator)
  ## Estimate the errors from the cov. matrix
  cts <- coeftest(mod, chc)

  if (length(var) == 0)
      return(cts)
  else
      return(cts[var,])
}

### Calculate clustered standard errors
## mod: the result of a call to lm()
##      e.g., lm(satell ~ width + factor(color))
## cluster: a cluster variable, with a length equal to the observation count
##      e.g., color
## var: a list of indexes to extract only certain coefficients (default: all)
##      e.g., 1:2 [to drop FE from above]
## Returns an object of type coeftest, with estimate, std. err, t and p values
get.coef.clust <- function(mod, cluster, var=c()) {
  require(sandwich, quietly = TRUE)
  require(lmtest, quietly = TRUE)

  ## Drop NA observations
  not <- attr(mod$model,"na.action")
  if (!is.null(not)) {
    cluster <- cluster[-not]
  }
  
  ## Drop unused factor levels
  if (is.factor(cluster)) {
    cluster <- droplevels(cluster)
  }
  
  ## Calculate the covariance matrix
  M <- length(unique(cluster))
  N <- length(cluster)
  K <- mod$rank
  dfc <- (M/(M-1))*((N-1)/(N-K))
  uj  <- apply(estfun(mod), 2, function(x) tapply(x, cluster, sum))
  vcovCL <- dfc*sandwich(mod, meat=crossprod(uj)/N)

  ## Estimate the errors from the cov. matrix
  cts <- coeftest(mod, vcovCL)

  if (length(var) == 0)
      return(cts)
  else
      return(cts[var,])
}

### Calculate confidence intervals for clustered standard errors
## mod: the result of a call to lm()
##      e.g., lm(satell ~ width + factor(color))
## cluster: a cluster variable, with a length equal to the observation count
##      e.g., color
## xx: new values for each of the variables (only needs to be as long as 'var')
##      e.g., seq(0, 50, length.out=100)
## alpha: the level of confidence, as an error rate
##      e.g., .05 [for 95% confidence intervals]
## var: a list of indexes to extract only certain coefficients (default: all)
##      e.g., 1:2 [to drop FE from above]
## Returns a data.frame of yhat, lo, and hi
get.conf.clust <- function(mod, cluster, xx, alpha, var=c()) {
    require(sandwich, quietly = TRUE)
    require(lmtest, quietly = TRUE)

    ## Drop NA observations
    not <- attr(mod$model,"na.action")
    if (!is.null(not)) {
        cluster <- cluster[-not]
    }
    
    ## Drop unused factor levels
    if (is.factor(cluster)) {
      cluster <- droplevels(cluster)
    }
    
    ## Calculate the covariance matrix
    M <- length(unique(cluster))
    N <- length(cluster)
    K <- mod$rank
    dfc <- (M/(M-1))*((N-1)/(N-K))
    uj  <- apply(estfun(mod), 2, function(x) tapply(x, cluster, sum))
    vcovCL <- dfc*sandwich(mod, meat=crossprod(uj)/N)

    ## Estimate new results, summing effects over variables
    results <- data.frame(yhat=c(), lo=c(), hi=c())
    for (ii in 1:nrow(xx)) {
        myxx <- rep(0, length(mod$coefficients))

        if (length(var) == 0)
            myxx <- xx[ii,]
        else
            myxx[var] <- xx[ii,]

        yhat <- sum(mod$coefficients * myxx, na.rm=T)
        myxxnn <- myxx[!is.na(mod$coefficients)]
        predscale <- as.numeric(sqrt(t(myxxnn) %*% vcovCL %*% myxxnn))
        hi <- yhat + abs(qt(alpha / 2, N - (K+1))) * predscale
        lo <- yhat - abs(qt(alpha / 2, N - (K+1))) * predscale

        results <- rbind(results, data.frame(yhat=yhat, lo=lo, hi=hi))
    }

    return(results)
}
