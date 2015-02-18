setwd("~/projects/dmas/trunk/lib/simple")

source("tools_reg.R")
tbl <- read.csv("data.csv")

mod <- lm(satell ~ width + factor(color), data=tbl)

summary(mod)
get.coef.robust(mod, estimator="HC2")
get.coef.clust(mod, tbl$color)

pooled <- function(means, sigmas) {
    variance <- 1 / sum(1 / sigmas^2)
    mean <- variance * sum(means / sigmas^2)
    return(c(mean, sqrt(variance)))
}

pooled(c(1, 2), c(1, 1))

pooled.weighted <- function(means, sigmas, weights) {
    variance <- 1 / sum(weights / sigmas^2)
    mean <- variance * sum(weights * means / sigmas^2)
    return(c(mean, sqrt(variance)))
}

pooled.weighted(c(1, 2), c(1, 1), c(1, .1))

### Treat as three different collections for Bayesian combination

setwd("~/projects/dmas/trunk/lib/simple")

source("tools_reg.R")
tbl <- read.csv("iris.csv")

library(ggplot2)

summary(lm(Sepal.length ~ Sepal.width + factor(Species), data=tbl))
summary(lm(Sepal.length ~ Sepal.width, data=subset(tbl, Species == "I. setosa")))
summary(lm(Sepal.length ~ Sepal.width, data=subset(tbl, Species == "I. versicolor")))
summary(lm(Sepal.length ~ Sepal.width, data=subset(tbl, Species == "I. virginica")))

species <- c("I. setosa", "I. versicolor", "I. virginica")
library(rstan)

#means <- c(0.3201, 0.3866, 0.5798)
#serrs <- c(0.2315, 0.3325, 0.1395)
means <- c(0.6905, 0.8651, 0.9015)
serrs <- c(0.0899, 0.2019, 0.2531)
pooled.vals <- pooled(means, serrs)
pooled.mean <- pooled.vals[1]
pooled.serr <- pooled.vals[2]

ggplot(tbl, aes(y=Sepal.length, x=Sepal.width, colour=Species, shape=Species)) +
    geom_point() + xlab("Sepal Width") + ylab("Sepal Length") +
    geom_abline(intercept=2.6390, slope=0.6905, colour=2) +
    geom_abline(intercept=3.5397, slope=0.8651, colour=3) +
    geom_abline(intercept=3.9068, slope=0.9015, colour=4)

schools_code <- '
  data {
    int<lower=0> J; // number of schools 
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates 
  }
  parameters {
    real mu; 
    real<lower=0> tau;
    real eta[J];
  }
  transformed parameters {
    real theta[J];
    for (j in 1:J)
      theta[j] <- mu + tau * eta[j];
  }
  model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
  }
'

schools_dat <- list(J = 3, y = means, sigma = serrs)

fit <- stan(model_code = schools_code, data = schools_dat, 
            iter = 2000, chains = 8)
la <- extract(fit, permuted=T)

##plot(density(la$mu))
##lines(density(rnorm(length(la$mu), la$mu, la$tau)))
spread <- rnorm(length(la$mu), la$mu, la$tau)

medians <- c(pooled.mean, means, quantile(la$theta[,1], .5), quantile(la$theta[,2], .5), quantile(la$theta[,3], .5), quantile(la$mu, .5), quantile(spread, .5))
lowers <- c(pooled.mean + qnorm(.25)*pooled.serr, means + qnorm(.25)*serrs, quantile(la$theta[,1], .25), quantile(la$theta[,2], .25), quantile(la$theta[,3], .25), quantile(la$mu, .25), quantile(spread, .25))
uppers <- c(pooled.mean + qnorm(.75)*pooled.serr, means + qnorm(.75)*serrs, quantile(la$theta[,1], .75), quantile(la$theta[,2], .75), quantile(la$theta[,3], .75), quantile(la$mu, .75), quantile(spread, .75))
ymins <- c(pooled.mean + qnorm(.1)*pooled.serr, means + qnorm(.1)*serrs, quantile(la$theta[,1], .1), quantile(la$theta[,2], .1), quantile(la$theta[,3], .1), quantile(la$mu, .1), quantile(spread, .1))
ymaxs <- c(pooled.mean + qnorm(.9)*pooled.serr, means + qnorm(.9)*serrs, quantile(la$theta[,1], .9), quantile(la$theta[,2], .9), quantile(la$theta[,3], .9), quantile(la$mu, .9), quantile(spread, .9))

specie <- factor(c("Pooled", rep(species, 2), "Bayes Hyper", "Bayes Spread"), levels=c(species, "Bayes Hyper", "Bayes Spread", "Pooled"), ordered=T)
facets <- factor(c("Pooled", rep("Independent", 3), rep("Partially Pooled", 5)), levels=c("Independent", "Partially Pooled", "Pooled"), ordered=T)
colors <- factor(c(1, 2:4, 2:4, 1, 1))

data <- data.frame(spine=specie, median=medians, lower=lowers, upper=uppers, ymin=ymins, ymax=ymaxs, facet=facets, color=colors)

ggplot(data, aes(x=spine)) +
    geom_boxplot(aes(ymin=ymin, lower=lower, middle=median, upper=upper, ymax=ymax, fill=color), stat='identity') +
    facet_grid(. ~ facet, scales="free_x", space="free") +
    guides(fill=FALSE) + xlab("Species")

### Add, remove, and reweight values

## Display original

schools_dat <- list(J = 3, y = means, sigma = serrs)

fit <- stan(fit=fit, data = schools_dat, 
            iter = 2000, chains = 8)
la <- extract(fit, permuted=T)

pooled.vals <- pooled(means, serrs)
pooled.mean <- pooled.vals[1]
pooled.serr <- pooled.vals[2]

medians <- c(pooled.mean, means, quantile(la$mu, .5))
lowers <- c(pooled.mean + qnorm(.25)*pooled.serr, means + qnorm(.25)*serrs, quantile(la$mu, .25))
uppers <- c(pooled.mean + qnorm(.75)*pooled.serr, means + qnorm(.75)*serrs, quantile(la$mu, .75))
ymins <- c(pooled.mean + qnorm(.1)*pooled.serr, means + qnorm(.1)*serrs, quantile(la$mu, .1))
ymaxs <- c(pooled.mean + qnorm(.9)*pooled.serr, means + qnorm(.9)*serrs, quantile(la$mu, .9))

specie <- factor(c("Pooled", species, "Bayes Hyper"), levels=c(species, "Bayes Hyper", "Bayes Spread", "Pooled"), ordered=T)
facets <- factor(c("(Partially) Pooled", rep("Independent", 3), "(Partially) Pooled"), levels=c("Independent", "(Partially) Pooled", "Pooled"), ordered=T)
colors <- factor(c(1, 2:4, 1))

data <- data.frame(spine=specie, median=medians, lower=lowers, upper=uppers, ymin=ymins, ymax=ymaxs, facet=facets, color=colors)

ggplot(data, aes(x=spine)) +
    geom_boxplot(aes(ymin=ymin, lower=lower, middle=median, upper=upper, ymax=ymax, fill=color), stat='identity') +
    facet_grid(. ~ facet, scales="free_x") +
    guides(fill=FALSE) + xlab("Sepal width-length Relationship Estimate") + scale_fill_manual(values=c("#f8766d", "#a3a500", "#02bf7d", "#01b0f6"))

# Display with addition

newserr <- .025
useserr <- sqrt(newserr^2 / .1) #.025
newmean <- 1.5 #pooled(means, serrs)[1]
schools_dat <- list(J = 4, y = c(means, newmean), sigma = c(serrs, useserr))

fit <- stan(fit=fit, data = schools_dat, 
            iter = 2000, chains = 8)
la <- extract(fit, permuted=T)

pooled.vals <- pooled(c(means, newmean), c(serrs, useserr))
pooled.mean <- pooled.vals[1]
pooled.serr <- pooled.vals[2]

medians <- c(pooled.mean, means, newmean, quantile(la$mu, .5))
lowers <- c(pooled.mean + qnorm(.25)*pooled.serr, means + qnorm(.25)*serrs, newmean + newserr*qnorm(.25), quantile(la$mu, .25))
uppers <- c(pooled.mean + qnorm(.75)*pooled.serr, means + qnorm(.75)*serrs, newmean + newserr*qnorm(.75), quantile(la$mu, .75))
ymins <- c(pooled.mean + qnorm(.1)*pooled.serr, means + qnorm(.1)*serrs, newmean + newserr*qnorm(.1), quantile(la$mu, .1))
ymaxs <- c(pooled.mean + qnorm(.9)*pooled.serr, means + qnorm(.9)*serrs, newmean + newserr*qnorm(.9), quantile(la$mu, .9))

specie <- factor(c("Pooled", species, "New Study", "Bayes Hyper"), levels=c(species, "Bayes Hyper", "Bayes Spread", "Pooled", "New Study"), ordered=T)
facets <- factor(c("(Partially) Pooled", rep("Independent", 4), "(Partially) Pooled"), levels=c("Independent", "(Partially) Pooled", "Pooled"), ordered=T)
colors <- factor(c(1, 2:5, 1))

data <- data.frame(spine=specie, median=medians, lower=lowers, upper=uppers, ymin=ymins, ymax=ymaxs, facet=facets, color=colors)

if (newserr == useserr) {
    ggplot(data, aes(x=spine)) +
        geom_boxplot(aes(ymin=ymin, lower=lower, middle=median, upper=upper, ymax=ymax, fill=color), stat='identity') +
            facet_grid(. ~ facet, scales="free_x") +
                guides(fill=FALSE) + theme(legend.position="none") +
                    xlab("Sepal width-length Relationship Estimate")
} else {
    alphas <- c(1, 1, 1, .1, 1, 1)

    ggplot(data, aes(x=spine)) +
        geom_boxplot(aes(ymin=ymin, lower=lower, middle=median, upper=upper, ymax=ymax, fill=color, alpha=alphas), stat='identity') +
            facet_grid(. ~ facet, scales="free_x") +
                guides(fill=FALSE) + theme(legend.position="none") +
                    xlab("Sepal width-length Relationship Estimate")
}    

### Construct a distribution

xxs <- seq(0, 2, length.out=1000)
ddp1 <- dnorm(xxs, means[1], serrs[1])
ddp2 <- dnorm(xxs, means[2], serrs[2])
ddp3 <- dnorm(xxs, means[3], serrs[3])

garden1 <- c(.5, .25, .25)
garden2 <- c(.25, .25, .5)

irises <- rbind(ddp1, ddp2, ddp3)
gardens <- rbind(garden1, garden2)

results <- t(irises) %*% t(gardens)

distdata <- data.frame(garden=c(rep("Garden 1", 3), rep("Garden 2", 3)), species=c("I. setosa", "I. versicolor", "I. virginica", "I. setosa", "I. versicolor", "I. virginica"), portion=c(garden1, garden2))

ggplot(distdata, aes(garden, weight=portion, fill=species)) +
  geom_bar() + ylab("Portion for Species") + xlab("") + scale_fill_manual(name="Species", values=c("#a3a500", "#02bf7d", "#01b0f6"))

edf1 <- cumsum(results[,1]) / sum(results[,1])
edf2 <- cumsum(results[,2]) / sum(results[,2])

medians <- xxs[c(min(which(edf1 > .5)), min(which(edf2 > .5)))]
lowers <- xxs[c(min(which(edf1 > .25)), min(which(edf2 > .25)))]
uppers <- xxs[c(min(which(edf1 > .75)), min(which(edf2 > .75)))]
ymins <- xxs[c(min(which(edf1 > .1)), min(which(edf2 > .1)))]
ymaxs <- xxs[c(min(which(edf1 > .9)), min(which(edf2 > .9)))]

data <- data.frame(garden=c("Garden 1", "Garden 2"), median=medians, lower=lowers, upper=uppers, ymin=ymins, ymax=ymaxs)

ggplot(data, aes(x=garden)) +
    geom_boxplot(aes(ymin=ymin, lower=lower, middle=median, upper=upper, ymax=ymax), stat='identity') +
    guides(fill=FALSE) + xlab("")
