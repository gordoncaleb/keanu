data { int<lower=1> T; vector[T] y; }
parameters { array[2] simplex[2] rows; ordered[2] means; real<lower=0> sigma; }
transformed parameters {
  matrix[2, 2] transition;
  for (i in 1:2) transition[i] = rows[i]';
}
model {
  means[1] ~ normal(0, 5);
  (means[2] - means[1]) ~ normal(0, 5);
  sigma ~ normal(0, 2);
  // rows ~ Dirichlet(1, 1): flat
  matrix[2, T] log_omega;
  for (t in 1:T)
    for (k in 1:2)
      log_omega[k, t] = normal_lpdf(y[t] | means[k], sigma);
  target += hmm_marginal(log_omega, transition, [0.5, 0.5]');
}
