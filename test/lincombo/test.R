library(rstan)

stan.code <- '
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

schools_dat <- list(J = 8,
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

fit <- stan(model_code = stan.code, data = schools_dat,
            iter = 1000, chains = 4)


stan.code <- '
data {
  int<lower=0> J; // number of estimates
  real y[J]; // estimated effects
  real<lower=0> sigma[J]; // s.e. of effect estimates
  int<lower=0> K; // number of coefficients
  vector[K] portions[J];
}
parameters {
  row_vector[K] mu;
  row_vector<lower=0>[K] tau;
  real eta[J];
}
transformed parameters {
  real beta[J];
  for (j in 1:J)
    beta[j] <- mu * portions[j] + tau * portions[j] * eta[j];
}
model {
  eta ~ normal(0, 1);
  y ~ normal(beta, sigma);
}
'

stan.dat <- list(J=2, y=c(0, 1), sigma=c(1, 1),
                 K=2, portions=matrix(c(1, 0, .5, .5), 2, 2))

##stan.dat <- list(J=3, y=c(0, 1, 0), sigma=c(1, 1, 2),
##                 K=1, portions=matrix(c(1, 1, 1), 3, 1))

##fit <- stan(model_code = stan.code, data = stan.dat,
##            iter = 1000, chains = 4)

fit <- stan(fit=fit, data = stan.dat,
            iter = 1000, chains = 4)

la <- extract(fit, permute=T)
colMeans(la$mu)
