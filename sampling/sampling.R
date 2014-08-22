library(rstan)

sizes <- c(919, 301, 180, 295, 266, 597, 1406, 29315, 2546)
effects <- c(34.4 - 36, 4.7 - 5, 3.2 - 5.2, 4.1 - 5.2, 4.2 - 4.6, 13.5 - 18, 5.6 - 5.9, 4.9 - 5.7, 5.5 - 5.6)

stan.code <- '
data {
  int<lower=0> N;
  int<lower=0> sizes[N];
  real<lower=0> sqrtsizes[N];
  real effects[N];
}
parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> sigma;
  real thetas[N];
}
model {
  thetas ~ normal(mu, tau);
  for (ii in 1:N)
    effects[ii] ~ student_t(sizes[ii], thetas[ii], sigma / sqrtsizes[ii]);
  //(effects - thetas) / (sigma / sqrtsizes) ~ student_t(sizes);
}
'

stan.dat <- list(N = length(sizes), sizes = sizes, effects = effects, sqrtsizes = sqrt(sizes))

fit <- stan(model_code = stan.code, data = stan.dat)

