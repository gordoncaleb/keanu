data {
  int<lower=1> N; int<lower=1> N_edges;
  array[N_edges] int<lower=0, upper=N-1> node1; array[N_edges] int<lower=0, upper=N-1> node2;
  real<lower=0> scaling_factor;
  vector<lower=0>[N] E; vector[N] x; array[N] int<lower=0> y;
}
transformed data {
  array[N_edges] int n1; array[N_edges] int n2;
  for (e in 1:N_edges) { n1[e] = node1[e] + 1; n2[e] = node2[e] + 1; }
  vector[N] log_E = log(E);
}
parameters { real beta0; real beta1; real<lower=0> sigma; real<lower=0, upper=1> rho; sum_to_zero_vector[N] phi; vector[N] theta; }
transformed parameters {
  vector[N] effect = sigma * (sqrt(rho / scaling_factor) * phi + sqrt(1 - rho) * theta);
}
model {
  beta0 ~ std_normal();
  beta1 ~ std_normal();
  sigma ~ std_normal();
  // rho ~ beta(1, 1): flat
  target += -0.5 * dot_self(phi[n1] - phi[n2]);
  theta ~ std_normal();
  y ~ poisson_log(log_E + beta0 + beta1 * x + effect);
}
