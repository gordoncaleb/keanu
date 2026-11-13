data { int<lower=1> N; vector[N] x; vector[N] y; }
parameters { real intercept; real slope; real<lower=0> scale; real<lower=0> nu; }
model {
  intercept ~ normal(0, 1);
  slope ~ normal(0, 2);
  scale ~ lognormal(log(0.5), 0.7);
  nu ~ lognormal(log(5.0), 0.9);
  y ~ student_t(nu, intercept + slope * x, scale);
}
