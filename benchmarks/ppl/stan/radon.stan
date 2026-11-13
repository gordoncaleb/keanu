data { int<lower=1> N; int<lower=1> J; array[N] int<lower=0, upper=J-1> county; vector[N] floor; vector[N] log_radon; }
transformed data { array[N] int idx; for (n in 1:N) idx[n] = county[n] + 1; }
parameters { real mu_a; real<lower=0> sigma_a; real b; real<lower=0> sigma_y; vector[J] a; }
model {
  mu_a ~ normal(0, 10);
  sigma_a ~ exponential(1);
  b ~ normal(0, 10);
  sigma_y ~ exponential(1);
  a ~ normal(mu_a, sigma_a);
  log_radon ~ normal(a[idx] + b * floor, sigma_y);
}
