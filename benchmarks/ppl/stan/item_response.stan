data { int<lower=1> N; int<lower=1> S; int<lower=1> I; array[N] int student; array[N] int item; array[N] int<lower=0, upper=1> correct; }
transformed data { array[N] int s1; array[N] int i1; for (n in 1:N) { s1[n] = student[n] + 1; i1[n] = item[n] + 1; } }
parameters {
  vector[S] ability;
  real difficulty_mean; real<lower=0> difficulty_scale; vector[I] difficulty_offset;
  real log_discrimination_mean; real<lower=0> log_discrimination_scale; vector[I] discrimination_offset;
}
transformed parameters {
  vector[I] difficulty = difficulty_mean + difficulty_scale * difficulty_offset;
  vector[I] discrimination = exp(log_discrimination_mean + log_discrimination_scale * discrimination_offset);
}
model {
  ability ~ std_normal();
  difficulty_mean ~ std_normal();
  difficulty_scale ~ lognormal(0, 0.5);
  difficulty_offset ~ std_normal();
  log_discrimination_mean ~ normal(0, 0.4);
  log_discrimination_scale ~ lognormal(log(0.3), 0.5);
  discrimination_offset ~ std_normal();
  correct ~ bernoulli_logit(discrimination[i1] .* (ability[s1] - difficulty[i1]));
}
