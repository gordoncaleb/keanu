data { int<lower=1> T; vector[T] r; }
parameters { real mu; real<lower=0, upper=1> phi_raw; real<lower=0> sigma; real eps_0; vector[T] eps; }
transformed parameters {
  real phi = 2 * phi_raw - 1;
  vector[T] h;
  {
    real prev = mu + sigma / sqrt(1 - phi * phi) * eps_0;
    for (t in 1:T) {
      prev = mu * (1 - phi) + phi * prev + sigma * eps[t];
      h[t] = prev;
    }
  }
}
model {
  mu ~ normal(0, 10);
  phi_raw ~ beta(20, 1.5);
  sigma ~ cauchy(0, 5);
  eps_0 ~ std_normal();
  eps ~ std_normal();
  r ~ normal(0, exp(h / 2));
}
