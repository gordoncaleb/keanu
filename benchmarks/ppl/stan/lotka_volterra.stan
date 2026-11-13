functions {
  vector lv(real t, vector z, vector rates) {
    vector[2] dz;
    dz[1] = z[1] * (rates[1] - rates[2] * z[2]);
    dz[2] = z[2] * (rates[4] * z[1] - rates[3]);
    return dz;
  }
}
data { int<lower=1> T; array[T] real ts; vector[2] y0; array[T] vector[2] y; real<lower=0> noise; }
parameters { vector<lower=0>[4] rates; }
model {
  rates ~ lognormal(log([0.8, 0.6, 0.6, 0.5]'), 0.35);
  array[T] vector[2] z = ode_rk45_tol(lv, y0, 0, ts, 1e-6, 1e-6, 1000000, rates);
  for (t in 1:T) y[t] ~ lognormal(log(z[t]), noise);
}
