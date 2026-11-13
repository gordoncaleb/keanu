data { int<lower=1> N; array[N] real x; vector[N] y; }
parameters { real<lower=0> amplitude; real<lower=0> length_scale; real<lower=0> noise; }
model {
  matrix[N, N] K = gp_exp_quad_cov(x, amplitude, length_scale);
  K = add_diag(K, square(noise) + 1e-6);
  amplitude ~ normal(0, 2);
  length_scale ~ normal(0, 2);
  noise ~ normal(0, 1);
  y ~ multi_normal_cholesky(rep_vector(0, N), cholesky_decompose(K));
}
