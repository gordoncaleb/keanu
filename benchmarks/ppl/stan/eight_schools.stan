data { int<lower=0> J; vector[J] y; vector<lower=0>[J] sigma; }
parameters { real mu; real<lower=0> tau; vector[J] eta; }
transformed parameters { vector[J] theta = mu + tau * eta; }
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  eta ~ std_normal();
  y ~ normal(theta, sigma);
}
