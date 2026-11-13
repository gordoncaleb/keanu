data { int<lower=1> N; vector[N] y; }
parameters { ordered[2] means; vector<lower=0>[2] sigmas; simplex[2] weights; }
model {
  // the same prior as Keanu's: first ~ N(0, 10), gap ~ HalfN(10) (the ordered transform's Jacobian is the identity's)
  means[1] ~ normal(0, 10);
  (means[2] - means[1]) ~ normal(0, 10);
  sigmas ~ normal(0, 5);
  weights ~ dirichlet([2, 2]');
  vector[2] lw = log(weights);
  for (n in 1:N)
    target += log_sum_exp(lw[1] + normal_lpdf(y[n] | means[1], sigmas[1]),
                          lw[2] + normal_lpdf(y[n] | means[2], sigmas[2]));
}
